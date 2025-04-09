#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import logging
import os
import re
import ssl
import sys
import traceback # Required for detailed error reporting
import winrm # Requires: pip install pywinrm
from winrm.exceptions import WinRMTransportError, WinRMOperationTimeoutError, WinRMError

# --- Color Class for Output ---
class Colors:
    """ Simple class for adding color codes to terminal output """
    try: is_tty = sys.stdout.isatty() and sys.stderr.isatty() # Check both streams
    except: is_tty = False
    if is_tty:
        HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'
        YELLOW = '\033[93m'; RED = '\033[91m'; BOLD = '\033[1m'
        UNDERLINE = '\033[4m'; ENDC = '\033[0m'
    else: HEADER = BLUE = GREEN = YELLOW = RED = BOLD = UNDERLINE = ENDC = ""
C = Colors()

# --- Basic Logging Setup ---
# Logger will be configured in main() after parsing args for verbosity
log = logging.getLogger(__name__)

# --- Parsing Functions (Adapted from local script + logging) ---
# Includes: hex_dump, decode_utf16le, find_text_start, clean_segment_text,
# extract_ascii_strings, find_pattern_at, parse_operations_PATTERN,
# apply_edits_PATTERN, extract_text_via_pattern_fallback

def hex_dump(data, bytes_per_line=16):
    """Generates a hex dump string for debugging."""
    output = []
    hex_part = ''
    text_part = ''
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]
        hex_part = ' '.join(f'{b:02X}' for b in chunk)
        text_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        output.append(f'{i:08X}: {hex_part.ljust(bytes_per_line * 3)}  {text_part}')
    return '\n'.join(output)

def decode_utf16le(data):
    """Decodes bytes as UTF-16LE, ignoring errors and stripping null terminators."""
    try:
        text = data.decode('utf-16le', errors='ignore')
        return text.rstrip('\x00')
    except Exception as e:
        log.error(f"{C.RED}[UTF-16LE Decode Error: {e}]{C.ENDC}")
        return f"[UTF-16LE Decode Error: {e}]"

def find_text_start(data, initial_offset=0, min_length=2, verbose_level=0):
    """Heuristically finds the start of a plausible UTF-16LE text block."""
    if initial_offset + (min_length * 2) > len(data):
        if verbose_level >= 2: log.debug(f"      [find_text_start: Data too short from offset {initial_offset}]")
        return -1
    for offset in range(initial_offset, len(data) - (min_length * 2) + 1):
        is_potential_start = True
        for i in range(min_length):
            char_byte_offset = offset + i * 2
            if char_byte_offset + 1 >= len(data): is_potential_start = False; break
            low_byte = data[char_byte_offset]; high_byte = data[char_byte_offset + 1]
            is_valid_char = (high_byte == 0x00)
            if not is_valid_char: is_potential_start = False; break
        if is_potential_start:
            if verbose_level >= 2: log.debug(f"      [find_text_start: Found plausible start at offset {offset}]")
            return offset
    if verbose_level >= 2: log.debug(f"      [find_text_start: No plausible start found from offset {initial_offset}]")
    return -1

def clean_segment_text(text):
    """Cleans extracted text by stripping whitespace and trailing non-printables."""
    cleaned = text.strip();
    while cleaned and (ord(cleaned[-1]) < 32 or 127 <= ord(cleaned[-1]) <= 159):
        cleaned = cleaned[:-1]
    return cleaned

def extract_ascii_strings(data, min_length=4):
    """Extracts consecutive ASCII printable/whitespace strings."""
    strings = []; current_string = ""
    for byte in data:
        if 32 <= byte <= 126 or byte in (9, 10, 13):
            current_string += chr(byte)
        else:
            if len(current_string) >= min_length: strings.append(current_string)
            current_string = ""
    if len(current_string) >= min_length: strings.append(current_string)
    return "\n".join(s.replace('\r\n', '\n').replace('\r', '\n') for s in strings)

def find_pattern_at(data_slice, index, verbose_level=0):
    """Checks for ADD/DELETE patterns. Debug prints only at level 2+."""
    slice_len = len(data_slice)
    if verbose_level >= 2:
        try: debug_bytes_len = min(4, slice_len - index); debug_bytes = data_slice[index:index+debug_bytes_len]; log.debug(f"    [DEBUG find_pattern_at] Idx={index:03d}, Check: {debug_bytes.hex(' ').upper()}")
        except Exception as e: log.debug(f"    [DEBUG find_pattern_at] Idx={index:03d}, Error: {e}")
    # Check ADD: pos, 0x00, 0x01, char
    if index + 4 <= slice_len:
        pos, b1, b2, char = data_slice[index:index+4]; is_add_match = (b1 == 0x00 and b2 == 0x01)
        if verbose_level >= 2: log.debug(f"        ADD Check: Bytes={pos:02X} {b1:02X} {b2:02X} {char:02X} -> Match? {is_add_match}")
        if is_add_match:
            if verbose_level >= 2: log.debug(f"        -> {C.GREEN}ADD MATCH!{C.ENDC}")
            return ('add', pos, char, 4)
    # Check DELETE: pos, 0x01, 0x00
    if index + 3 <= slice_len:
        pos, b1, b2 = data_slice[index:index+3]; is_del_match = (b1 == 0x01 and b2 == 0x00)
        if verbose_level >= 2: log.debug(f"        DEL Check: Bytes={pos:02X} {b1:02X} {b2:02X} -> Match? {is_del_match}")
        if is_del_match:
            if verbose_level >= 2: log.debug(f"        -> {C.RED}DEL MATCH!{C.ENDC}")
            return ('delete', pos, None, 3)
    if verbose_level >= 2: log.debug(f"        -> NO MATCH.")
    return None

def parse_operations_PATTERN(data_slice, base_text_length=0, verbose_level=0):
    """UNIFIED PARSER: Checks every byte offset, always advances index by 1."""
    found_ops_relative_temp = []; current_index = 0; slice_len = len(data_slice)
    if verbose_level >= 2: log.debug(f"{C.BLUE}--- Parsing Operations (ALWAYS ADVANCE 1 Mode) ---{C.ENDC}")
    while current_index < slice_len:
        result = find_pattern_at(data_slice, current_index, verbose_level=verbose_level)
        if result: op_type, pos, char_code, length = result; found_ops_relative_temp.append({'index': current_index, 'type': op_type, 'pos': pos, 'char': char_code}); log.debug(f"  Idx={current_index:03d}: {C.BOLD}RECORDING:{C.ENDC} {op_type}, Pos={pos:02X}, Len={length}") if verbose_level >= 2 else None
        current_index += 1
    if verbose_level >= 2: log.debug(f"{C.BLUE}--- Pattern Scan Complete ---{C.ENDC}")
    final_ops_relative = [(op['index'], op['type'], op['pos'], op['char']) for op in found_ops_relative_temp]; final_ops_relative.sort(key=lambda x: x[0])
    if verbose_level >= 2: log.debug(f"{C.YELLOW}--- Rule 4 Validation SKIPPED ---{C.ENDC}")
    if verbose_level >= 2:
        log.debug(f"{C.BLUE}--- Final Pattern Operations List (Unfiltered) ---{C.ENDC}")
        if not final_ops_relative: log.debug("   (None)")
        else:
            max_ops_to_print_parser = 100; op_count = 0
            for op in final_ops_relative:
                op_color = C.GREEN if op[1] == 'add' else C.RED if op[1] == 'delete' else C.ENDC
                log.debug(f"   (RelIdx={op[0]:03d}, {op_color}{op[1]:<6}{C.ENDC}, Pos=0x{op[2]:02X}, CharCode={f'0x{op[3]:02X}' if op[3] is not None else 'None'})")
                op_count += 1
                if op_count >= max_ops_to_print_parser:
                    log.debug(f"      ... (list truncated)")
                    break # Correctly indented
    return final_ops_relative

def apply_edits_PATTERN(operations, initial_text="", verbose_level=0):
    """UNIFIED APPLY LOGIC: Applies ADD/DELETE operations assuming INSERT/SHIFT behavior."""
    text_list = list(initial_text); deleted_chars_list = []
    if verbose_level >= 2:
        log.debug(f"{C.HEADER}--- Applying Edits (INSERT/SHIFT Mode - Detailed Trace) ---{C.ENDC}")
        if operations:
            log.debug(f"    {C.BLUE}Operations to Apply ({len(operations)}):{C.ENDC}")
            max_ops_to_print = 50; op_count = 0
            operations.sort(key=lambda op: op[0])
            for op in operations:
                 op_index, op_type, op_pos, op_char_code = op
                 op_char_repr = f'0x{op_char_code:02X}' if op_char_code is not None else 'None'
                 op_color = C.GREEN if op_type == 'add' else C.RED if op_type == 'delete' else C.ENDC
                 log.debug(f"     - OrigIdx={op_index:03d}, Type={op_color}{op_type:<6}{C.ENDC}, Pos=0x{op_pos:02X}, CharCode={op_char_repr}")
                 op_count += 1
                 if op_count >= max_ops_to_print:
                     log.debug(f"     ... (list truncated)")
                     break # Correctly indented
        else: log.debug(f"    {C.BLUE}Operations to Apply:{C.ENDC} (None)")
        log.debug(f"{C.BLUE}--- Beginning Detailed Edit Application ---{C.ENDC}")

    if not operations: return initial_text
    operations.sort(key=lambda op: op[0])
    for index, op_type, position, char_code in operations:
        char_repr = None; actual_char = None
        if op_type == 'add' and char_code is not None:
            try:
                if 0 <= char_code <= 127: actual_char = chr(char_code); char_repr = actual_char if actual_char.isprintable() or actual_char in ('\n', '\r', '\t') else f'\\x{char_code:02x}'
                else: actual_char = bytes([char_code]).decode('utf-8', errors='replace'); char_repr = actual_char if actual_char.isprintable() or actual_char in ('\n', '\r', '\t') else '?'; char_repr = '?' if char_repr == '\ufffd' else char_repr
            except Exception as decode_err: char_repr = f'\\x{char_code:02x}'; actual_char = None; log.debug(f"    Err interpreting char 0x{char_code:02X}: {decode_err}") if verbose_level >= 2 else None
        if verbose_level >= 2: current_len=len(text_list); op_color = C.GREEN if op_type == 'add' else C.RED if op_type == 'delete' else C.ENDC; log.debug(f"    Op @ Idx {index:03d}: {op_color}{op_type}{C.ENDC}, Pos=0x{position:02X}, Char='{char_repr if char_repr is not None else 'N/A'}', Code=({f'0x{char_code:02X}' if char_code is not None else 'N/A'}), CurLen={current_len}")
        try:
            if op_type == 'add':
                insert_pos = max(0, min(position, len(text_list)))
                if actual_char is not None: text_list.insert(insert_pos, actual_char); log.debug(f"     -> {C.GREEN}Inserted AT index {insert_pos}. New len={len(text_list)}{C.ENDC}") if verbose_level >= 2 else None
                elif verbose_level >= 2: log.debug(f"     -> {C.YELLOW}Add op with uninterpretable/None char skipped.{C.ENDC}")
            elif op_type == 'delete':
                delete_pos = position
                if 0 <= delete_pos < len(text_list): char_to_delete = text_list[delete_pos]; deleted_chars_list.append(char_to_delete); log.debug(f"     -> {C.RED}{C.BOLD}Deleted char: {repr(char_to_delete)} (from index {delete_pos}){C.ENDC}") if verbose_level >= 2 else None; text_list.pop(delete_pos); log.debug(f"     -> {C.RED}List len after delete: {len(text_list)}{C.ENDC}") if verbose_level >= 2 else None
                elif verbose_level >= 2: log.debug(f"     -> {C.YELLOW}Delete pos 0x{delete_pos:02X} ({delete_pos}) out of bounds (len={len(text_list)}). Skipped.{C.ENDC}")
        except Exception as e: log.error(f"{C.RED}ERROR applying op {op_type} @ {index} (Pos: {position}): {e}{C.ENDC}"); log.debug(traceback.format_exc(limit=1)) if verbose_level >= 1 else None
    final_text = "".join(text_list)
    # --- Edit Summary Logging (Level 1+), WITHOUT Final Text ---
    if verbose_level >= 1:
        log.info(f"{C.BLUE}--- Edit Summary ---{C.ENDC}")
        label_width = 28; log.info(f"  {C.GREEN}{'Initial Text':<{label_width}}{C.ENDC}: {repr(initial_text)}")
        deleted_chars_raw = "".join(deleted_chars_list); reversed_deleted_chars = deleted_chars_raw[::-1]; formatted_deleted_output = reversed_deleted_chars.replace('\r\n', '\n').replace('\r', '\n').strip()
        log.info(f"  {C.RED}{f'Deleted Characters ({len(deleted_chars_raw)} chars)':<{label_width}}{C.ENDC}:")
        deleted_lines = formatted_deleted_output.splitlines()
        if deleted_lines: first_line_repr = repr(deleted_lines[0]); log.info(f"      {first_line_repr}"); [log.info(f"      {repr(line)}") for line in deleted_lines[1:]]
        elif deleted_chars_raw: log.info(f"      (Whitespace/non-printable only: {repr(deleted_chars_raw)})")
        else: log.info(f"      (None)")
        # Final text is NOT logged here anymore
    if verbose_level >= 2: log.debug(f"{C.BLUE}--- Applying Edits Complete ---{C.ENDC}")
    return final_text

def extract_text_via_pattern_fallback(data_bytes, source_description="", verbose_level=0):
    """Applies the UNIFIED pattern parser and apply logic as a fallback mechanism."""
    if verbose_level >= 1: log.debug(f"  {C.BLUE}[Pattern Fallback attempting for {source_description}...]{C.ENDC}")
    if not data_bytes or len(data_bytes) < 2: return ""
    start_index = 2
    if start_index >= len(data_bytes): log.debug(f"  {C.YELLOW}[Pattern Fallback: Data too short from index {start_index}]{C.ENDC}") if verbose_level >= 1 else None; return ""
    data_to_parse = data_bytes[start_index:]; original_offset = start_index
    operations_relative = parse_operations_PATTERN(data_to_parse, base_text_length=0, verbose_level=verbose_level)
    operations = [(rel_idx + original_offset, op_type, op_pos, op_char) for rel_idx, op_type, op_pos, op_char in operations_relative]
    final_text = apply_edits_PATTERN(operations, initial_text="", verbose_level=verbose_level)
    return final_text

# process_saved_file returns dictionary
def process_saved_file(data, file_path_for_log, verbose_level=0):
    """Processes saved files using UNIFIED Logic, returns {'reconstructed': text} or None."""
    log.info(f"{C.BOLD}[Saved File - Using Unified Logic]{C.ENDC} ({file_path_for_log})")
    if verbose_level >= 1: log.debug("\nHex Dump (first 256 bytes):\n" + hex_dump(data[:256])); log.debug("...") if len(data)>256 else None

    file_path_start_offset = 5; max_path_scan_len = 512; scan_end_offset = min(file_path_start_offset + max_path_scan_len, len(data))
    file_path = "[Path Extraction Error]"; valid_path_bytes = bytearray()
    path_log_level = logging.DEBUG if verbose_level >= 1 else logging.INFO
    if file_path_start_offset < len(data) - 1:
        try:
            idx = file_path_start_offset
            while idx < scan_end_offset - 1:
                low, high = data[idx:idx+2];
                if low == 0 and high == 0: break
                if high != 0: log.debug(f"  {C.YELLOW}[Path Extractor: Non-zero high byte 0x{high:02X} @{idx+1}]{C.ENDC}") if verbose_level >= 1 else None; break
                valid_path_bytes.extend([low, high]); idx += 2
            file_path = decode_utf16le(valid_path_bytes) if valid_path_bytes else "[Path Extraction Failed]"
        except Exception as e: file_path = f"[Path Extraction Error: {e}]"; log.error(f"  {C.RED}Error path extraction: {e}{C.ENDC}"); log.debug(traceback.format_exc(limit=1)) if verbose_level >= 1 else None
        log.log(path_log_level, f"{C.BLUE}  File Path in Binary:{C.ENDC} {file_path}")
    else: log.warning(f"  {C.YELLOW}[Warning: Not enough data for path]{C.ENDC}")

    if verbose_level >= 1: log.debug(f"{C.BLUE}--- Searching for Initial Content Block ---{C.ENDC}");
    base_text = ""; end_base_text_idx = -1; base_text_marker = b'\x02\x01\x01'; base_text_found = False
    try:
        marker_idx = data.find(base_text_marker)
        if marker_idx != -1:
            len_byte_idx = marker_idx + len(base_text_marker)
            if len_byte_idx < len(data):
                length = data[len_byte_idx]; start_text_idx = len_byte_idx + 1; end_text_idx_calc = start_text_idx + length * 2
                if end_text_idx_calc <= len(data): base_text = decode_utf16le(data[start_text_idx : end_text_idx_calc]); end_base_text_idx = end_text_idx_calc; base_text_found = True; log.debug(f"  {C.GREEN}[Base Text Found: len={length}, raw='{repr(base_text)}']{C.ENDC}") if verbose_level >= 1 else None
                else: log.debug(f"  {C.YELLOW}[Base Text: End {end_text_idx_calc} > data len {len(data)}]{C.ENDC}") if verbose_level >= 1 else None
            else: log.debug(f"  {C.YELLOW}[Base Text: Found marker but no length byte]{C.ENDC}") if verbose_level >= 1 else None
        if not base_text_found: log.debug(f"  {C.YELLOW}[Base Text: Marker {base_text_marker.hex()} not found]{C.ENDC}") if verbose_level >= 1 else None
    except Exception as e: log.error(f"  {C.RED}[Error extracting base text: {e}]{C.ENDC}"); log.debug(traceback.format_exc(limit=1)) if verbose_level >= 1 else None

    base_text_len = len(base_text); base_text_log_level = logging.DEBUG if verbose_level >= 1 else logging.INFO
    if base_text_found: log.log(base_text_log_level, f"{C.GREEN}  Initial Base Text ({base_text_len} chars):{C.ENDC}\n    '{base_text.replace('\r', '\n')}'")
    else: log.warning(f"{C.YELLOW}  [No Initial Base Text Found]{C.ENDC}"); end_base_text_idx = 2

    if verbose_level >= 1: log.debug(f"{C.BLUE}--- Parsing/Applying Operations (From index {end_base_text_idx}) ---{C.ENDC}");
    final_reconstructed_text = None; error_occurred = False
    try:
        data_to_parse_ops = data[end_base_text_idx:] if end_base_text_idx < len(data) else b''; original_offset = end_base_text_idx
        if not data_to_parse_ops:
            final_reconstructed_text = base_text
            log.debug(f"  {C.YELLOW}[Info: No operation data found. Using base text.]{C.ENDC}") if verbose_level >= 1 else None
            if verbose_level >= 1: apply_edits_PATTERN([], initial_text=base_text, verbose_level=verbose_level) # Trigger summary log
        else:
            operations_relative = parse_operations_PATTERN(data_to_parse_ops, base_text_length=base_text_len, verbose_level=verbose_level)
            operations = [(rel_idx + original_offset, op_type, op_pos, op_char) for rel_idx, op_type, op_pos, op_char in operations_relative]
            final_reconstructed_text = apply_edits_PATTERN(operations, initial_text=base_text, verbose_level=verbose_level)
    except Exception as pattern_error: log.error(f"{C.RED}[Error during saved file parsing/application: {pattern_error}]{C.ENDC}"); log.error(traceback.format_exc()); error_occurred = True

    return {'reconstructed': final_reconstructed_text} if not error_occurred else None

# process_unsaved_file returns dictionary with multiple results, NO internal result logging
def process_unsaved_file(data, file_path_for_log, verbose_level=0):
    """Processes unsaved files: Tries Primary, ASCII, PATTERN. Returns dict with results."""
    log.info(f"{C.BOLD}[Unsaved File]{C.ENDC} ({file_path_for_log})")
    if verbose_level >= 1: log.debug("\nHex Dump (first 256 bytes):\n" + hex_dump(data[:256])); log.debug("...") if len(data)>256 else None

    primary_extracted_text = None; ascii_fallback_text = None; pattern_fallback_text = None

    # --- Attempt 1: Primary Method ---
    if verbose_level >= 1: log.debug(f"{C.BLUE}--- Attempting Primary Method (UTF-16LE Heuristics) ---{C.ENDC}")
    text_start_offset = find_text_start(data, 0, min_length=2, verbose_level=verbose_level)
    separator_offset = -1
    try: separator_offset = data.rfind(b'\x01')
    except Exception as e: log.debug(f"  {C.YELLOW}[Warning: data.rfind(b'\\x01') failed: {e}]{C.ENDC}")
    if separator_offset != -1 and text_start_offset != -1 and text_start_offset < separator_offset:
        text_data = data[text_start_offset:separator_offset]
        if verbose_level >= 1: log.debug(f"  [Primary Method: Found start={text_start_offset}, sep={separator_offset}. Decoding {len(text_data)} bytes.]")
        decoded_text = decode_utf16le(text_data); cleaned_text = clean_segment_text(decoded_text)
        if cleaned_text: primary_extracted_text = cleaned_text # Store result
        else: log.debug(f"  {C.YELLOW}[Primary Method: UTF-16LE decoding yielded empty result.]{C.ENDC}") if verbose_level >= 1 else None
    elif verbose_level >= 1: log.debug(f"  {C.YELLOW}[Primary Method: Heuristics failed.]{C.ENDC}")

    # --- Attempt 2: ASCII Fallback ---
    if verbose_level >= 1: log.debug(f"{C.YELLOW}--- Attempting ASCII Fallback ---{C.ENDC}")
    try:
        ascii_text = extract_ascii_strings(data)
        if ascii_text: ascii_fallback_text = ascii_text # Store result
        elif verbose_level >= 1: log.debug(f"  {C.YELLOW}[No Content Extracted via ASCII Fallback]{C.ENDC}")
    except Exception as ascii_error: log.error(f"  {C.RED}[ASCII Fallback Error: {ascii_error}]{C.ENDC}"); log.debug(traceback.format_exc(limit=1)) if verbose_level >= 1 else None

    # --- Attempt 3: PATTERN PARSER Fallback ---
    if verbose_level >= 1: log.debug(f"{C.BLUE}--- Attempting PATTERN PARSER Fallback ---{C.ENDC}")
    try:
        pattern_text = extract_text_via_pattern_fallback(data, file_path_for_log, verbose_level=verbose_level)
        if pattern_text is not None: pattern_fallback_text = pattern_text # Store result
        elif verbose_level >= 1: log.debug(f"  {C.YELLOW}[PATTERN PARSER Fallback returned None/Error]{C.ENDC}")
    except Exception as pattern_error: log.error(f"{C.RED}[Pattern Parser Fallback Error: {pattern_error}]{C.ENDC}"); log.error(traceback.format_exc())

    # Return dictionary containing all results (they can be None)
    # NO internal logging of the results here - main loop handles printing
    return {'primary': primary_extracted_text, 'pattern': pattern_fallback_text, 'ascii': ascii_fallback_text}

# --- Remote Execution Function ---
def execute_remote_ps(session, command):
    """Executes PowerShell command via WinRM, returns stdout, stderr."""
    log.debug(f"Executing PS: {command}")
    try:
        if isinstance(command, list): command = ' '.join(command)
        result = session.run_ps(command)
        stdout = result.std_out.decode('utf-8', errors='ignore').strip() if result.std_out else ""
        stderr = result.std_err.decode('utf-8', errors='ignore').strip() if result.std_err else ""
        if result.status_code != 0:
            log.warning(f"{C.YELLOW}PS command exited with code {result.status_code}{C.ENDC}")
            error_msg = stderr if stderr else stdout; error_msg = error_msg if error_msg else f"PS command failed (Code: {result.status_code})"
            log.debug(f"PS Error Output: {error_msg}")
            if "Cannot find path" in error_msg: log.warning(f"  {C.YELLOW}Reason: Path not found/inaccessible.{C.ENDC}")
            elif "Access is denied" in error_msg or "AuthorizationManager" in error_msg : log.warning(f"  {C.YELLOW}Reason: Access denied (Permissions?).{C.ENDC}")
            elif "rejected by the server" in error_msg: log.warning(f"  {C.YELLOW}Reason: Credentials rejected.{C.ENDC}")
            return stdout, error_msg
        log.debug(f"PS Stdout: {stdout}"); log.debug(f"PS Stderr: {stderr}") if stderr else None
        return stdout, stderr
    except (WinRMTransportError, WinRMOperationTimeoutError, WinRMError) as e:
        log.error(f"{C.RED}WinRM error: {e}{C.ENDC}"); log.error(f"  {C.RED}Detail: Authentication failed - check credentials/permissions.{C.ENDC}") if "rejected" in str(e) else None
        return None, str(e)
    except Exception as e: log.error(f"{C.RED}Unexpected PS exec error: {e}{C.ENDC}"); log.debug(traceback.format_exc()); return None, str(e)

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description=f"{C.HEADER}Remotely parse Notepad TabState files via WinRM.{C.ENDC}", epilog=f"Examples:\n  {os.path.basename(sys.argv[0])} --host HOST -u USER -p PASS\n  {os.path.basename(sys.argv[0])} --host HOST -u ADMIN -p ADMPASS --target-user OTHERUSER -vv", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--host", required=True, help="Target hostname or IP.")
    parser.add_argument("-u", "--username", required=True, help="WinRM auth username.")
    parser.add_argument("-p", "--password", required=True, help="WinRM auth password.")
    parser.add_argument("--target-user", default=None, help="Target username profile (requires admin creds if different).")
    parser.add_argument("--port", type=int, default=None, help="WinRM port (default: 5985/http, 5986/https).")
    parser.add_argument("--transport", default="ntlm", choices=['basic', 'kerberos', 'ntlm', 'plaintext', 'ssl', 'negotiate', 'credssp'], help="WinRM transport protocol (default: ntlm).")
    parser.add_argument("--no-verify-ssl", action='store_true', help="Ignore SSL cert verification.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity: -v Debug, -vv Detail trace.")
    args = parser.parse_args()

    verbose_level = args.verbose
    log_level = logging.DEBUG if verbose_level >= 1 else logging.INFO
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s', stream=sys.stderr) # Log to stderr
    log.info(f"{C.HEADER}--- Notepad Remote Decoder ---{C.ENDC}")
    log.debug(f"{C.BLUE}Verbose logging enabled (Level {verbose_level}){C.ENDC}") if verbose_level >= 1 else None

    effective_target_user = args.username
    if args.target_user and args.target_user.lower() != args.username.lower():
        effective_target_user = args.target_user
        log.warning(f"{C.YELLOW}Targeting user: '{effective_target_user}'. Requires admin privileges.{C.ENDC}")
        path_param = "-LiteralPath"; ps_base_path_value = f"C:\\Users\\{effective_target_user}\\AppData\\Local\\Packages\\Microsoft.WindowsNotepad_8wekyb3d8bbwe\\LocalState\\TabState"
        conceptual_path_prefix = f"C:\\Users\\{effective_target_user}\\...\\TabState"
    else:
        log.info(f"Targeting profile for user: '{effective_target_user}'")
        path_param = "-Path"; ps_base_path_value = f"$env:LOCALAPPDATA\\Packages\\Microsoft.WindowsNotepad_8wekyb3d8bbwe\\LocalState\\TabState"
        conceptual_path_prefix = "%LOCALAPPDATA%\\...\\TabState"

    is_https = args.transport in ['ssl', 'credssp'] or args.port == 5986
    transport_protocol = 'https' if is_https else 'http'
    default_port = 5986 if is_https else 5985; port = args.port if args.port else default_port
    endpoint_target = f"{transport_protocol}://{args.host}:{port}/wsman"
    server_cert_validation = 'ignore' if args.no_verify_ssl or not is_https else 'validate'
    log.info(f"Connecting to {C.BLUE}{endpoint_target}{C.ENDC} as '{args.username}' via '{args.transport}'...")

    session = None; parsed_content_count = 0; found_files_count = 0
    try:
        session = winrm.Session(endpoint_target, auth=(args.username, args.password), transport=args.transport, server_cert_validation=server_cert_validation, read_timeout_sec=30, operation_timeout_sec=25)
        log.info(f"{C.GREEN}WinRM Session established.{C.ENDC}")

        ps_list_cmd = f"$ProgressPreference = 'SilentlyContinue'; Get-ChildItem {path_param} \"{ps_base_path_value}\" -Filter *.bin -EA SilentlyContinue | Select-Object -ExpandProperty Name"
        log.info(f"Listing files for '{effective_target_user}'..."); log.debug(f" List cmd: {ps_list_cmd}")
        file_list_output, list_error = execute_remote_ps(session, ps_list_cmd)
        list_error_str = str(list_error).strip(); file_list_output_str = str(file_list_output).strip()
        if list_error_str and not file_list_output_str: log.error(f"{C.RED}Failed list files: {list_error_str}{C.ENDC}"); sys.exit(1)
        if not file_list_output_str: log.info(f"  {C.YELLOW}No .bin files found.{C.ENDC}"); sys.exit(0)
        filenames = file_list_output_str.strip().splitlines(); log.debug(f" Found {len(filenames)} potential filenames.")

        log.info("-" * 40); log.info(f"Attempting read/parse for {len(filenames)} file(s)...")
        for filename in filenames:
            if not filename or not filename.endswith(".bin"): continue
            if re.match(r'.*\.\d+\.bin$', filename): log.debug(f"  Skipping history file: {filename}"); continue
            found_files_count += 1
            conceptual_remote_path = f"{conceptual_path_prefix}\\{filename}"
            log.info(f"\n{C.HEADER}>>> Processing file: {filename} for user '{effective_target_user}' <<< {C.ENDC}")

            ps_file_path = f"{ps_base_path_value}\\{filename}"
            ps_get_content_cmd = f"$ProgressPreference='SilentlyContinue';$ErrorActionPreference='Stop';$theFilePath=\"{ps_file_path}\";if(Test-Path {path_param} $theFilePath -EA SilentlyContinue){{try{{$bytes=Get-Content {path_param} $theFilePath -Enc Byte -Raw;if($bytes.Length -gt 0){{[Convert]::ToBase64String($bytes);}}else{{Write-Output 'EMPTY_FILE';}}}}catch{{$errMsg=$_.Exception.Message -replace '\"','''' -replace \"`n\",\" \" -replace \"`r\",\" \";Write-Output \"READ_ERROR:$errMsg\";}}}}else{{Write-Output 'FILE_NOT_FOUND';}}"
            log.debug(f" Getting content for: {filename}"); log.debug(f"  Get cmd: {ps_get_content_cmd}")
            base64_content, content_error = execute_remote_ps(session, ps_get_content_cmd)
            content_error_str = str(content_error).strip(); base64_content_str = str(base64_content).strip()
            if content_error_str and "READ_ERROR:" not in base64_content_str: log.error(f"  {C.RED}Get-Content cmd error: {content_error_str}{C.ENDC}"); continue
            if "READ_ERROR:" in base64_content_str: error_detail = base64_content_str.split("READ_ERROR:", 1)[-1].strip(); log.warning(f"  {C.YELLOW}PS read error: {error_detail}{C.ENDC}"); continue
            if "FILE_NOT_FOUND" in base64_content_str: log.warning(f"  {C.YELLOW}File not found: {filename}{C.ENDC}"); continue
            if "EMPTY_FILE" in base64_content_str: log.warning(f"  {C.YELLOW}File is empty: {filename}{C.ENDC}"); continue
            if not base64_content_str: log.warning(f"  {C.YELLOW}No content received: {filename}.{C.ENDC}"); continue
            try: file_content_bytes = base64.b64decode(base64_content_str)
            except Exception as decode_error: log.error(f"  {C.RED}Base64 decode error: {decode_error}{C.ENDC}"); continue
            log.debug(f"  Read/decoded {len(file_content_bytes)} bytes.")
            if len(file_content_bytes) < 4: log.warning(f"  {C.YELLOW}File content < 4 bytes. Skipping.{C.ENDC}"); continue

            is_saved = (file_content_bytes[3] == 1)
            results = None
            try:
                if is_saved: results = process_saved_file(file_content_bytes, conceptual_remote_path, verbose_level=verbose_level)
                else: results = process_unsaved_file(file_content_bytes, conceptual_remote_path, verbose_level=verbose_level)
            except Exception as processing_error: log.error(f"  {C.RED}Content processing error: {processing_error}{C.ENDC}"); log.error(traceback.format_exc()); continue

            # --- Print Results to STDOUT ---
            if results:
                content_found_in_file = False; output_lines = []
                if is_saved:
                    reconstructed = results.get('reconstructed')
                    if reconstructed is not None:
                        # Apply BOLD formatting to content
                        output_lines.append(f"{C.GREEN}[Reconstructed (Saved)]:{C.ENDC}\n{C.BOLD}{reconstructed}{C.ENDC}")
                        content_found_in_file = bool(reconstructed)
                else: # Unsaved file results dictionary
                    primary = results.get('primary'); pattern = results.get('pattern'); # ascii_res = results.get('ascii')
                    if primary is not None:
                         # Apply BOLD formatting to content
                        output_lines.append(f"{C.BLUE}[Primary UTF-16LE]:{C.ENDC}\n{C.BOLD}{primary}{C.ENDC}")
                        content_found_in_file = content_found_in_file or bool(primary)
                    if pattern is not None:
                         # Apply BOLD formatting to content
                        output_lines.append(f"{C.BLUE}[PATTERN Fallback]:{C.ENDC}\n{C.BOLD}{pattern}{C.ENDC}")
                        content_found_in_file = content_found_in_file or bool(pattern)
                    # if ascii_res is not None: # Optionally add ASCII output
                    #     output_lines.append(f"{C.YELLOW}[ASCII Fallback]:{C.ENDC}\n{C.BOLD}{ascii_res}{C.ENDC}")
                    #     content_found_in_file = content_found_in_file or bool(ascii_res)

                if output_lines:
                    if content_found_in_file: parsed_content_count += 1
                    # Print collected results to stdout, separated by double newline
                    print("\n\n".join(output_lines), file=sys.stdout); print("", file=sys.stdout)
                elif not content_found_in_file: log.info(f"  {C.YELLOW}No content extracted for {filename} by relevant methods.{C.ENDC}")
            else: log.warning(f"  {C.YELLOW}Processing function returned None for {filename}.{C.ENDC}")

    except (WinRMTransportError, WinRMOperationTimeoutError, WinRMError) as e: log.critical(f"{C.RED}FATAL WinRM Error: {e}{C.ENDC}"); log.critical(f"{C.RED}Check connection, credentials, permissions.{C.ENDC}")
    except Exception as e: log.critical(f"{C.RED}FATAL Error: {e}{C.ENDC}"); log.error(traceback.format_exc())
    finally: pass

    log.info("=" * 40)
    summary_user = f"profile for user '{effective_target_user}'"
    if parsed_content_count > 0: log.info(f"{C.GREEN}Finished. Checked {summary_user}. Found content in {parsed_content_count}/{found_files_count} relevant file(s).{C.ENDC}")
    elif found_files_count > 0: log.info(f"{C.YELLOW}Finished. Checked {summary_user}. Found {found_files_count} file(s), but no content extracted.{C.ENDC}")
    else: log.info(f"{C.YELLOW}Finished. Checked {summary_user}. No relevant files found/read.{C.ENDC}")
    log.info("=" * 40)

if __name__ == "__main__":
    main()