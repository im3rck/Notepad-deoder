#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import glob
import collections
import datetime
import sys
import traceback # For detailed error reporting

# Attempt to import pytz for timezone context, but don't fail if unavailable
try:
    import pytz
except ImportError:
    pytz = None

# --- Color Class for Output ---
class Colors:
    try: is_tty = sys.stdout.isatty()
    except: is_tty = False
    if is_tty:
        HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'
        YELLOW = '\033[93m'; RED = '\033[91m'; BOLD = '\033[1m'
        UNDERLINE = '\033[4m'; ENDC = '\033[0m'
    else: HEADER = BLUE = GREEN = YELLOW = RED = BOLD = UNDERLINE = ENDC = ""
C = Colors()

# --- Helper Functions (Unchanged) ---

def hex_dump(data, bytes_per_line=16):
    """Returns a formatted hex dump string of the given data."""
    output = []; hex_part = ''; text_part = ''
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]; hex_part = ' '.join(f'{b:02X}' for b in chunk)
        text_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        output.append(f'{i:08X}: {hex_part.ljust(bytes_per_line * 3)}  {text_part}')
    return '\n'.join(output)

def decode_utf16le(data):
    """Decodes bytes as UTF-16LE, stripping trailing nulls. Includes error handling."""
    try:
        text = data.decode('utf-16le', errors='ignore')
        return text.rstrip('\x00')
    except Exception as e:
        return f"[UTF-16LE Decode Error: {e}]"

def find_text_start(data, initial_offset=0, min_length=2, verbose=False):
    """Heuristic to find the start of plausible UTF-16LE text (ASCII range chars followed by 0x00)."""
    if initial_offset + (min_length * 2) > len(data):
        if verbose: print(f"      [find_text_start: Data too short from offset {initial_offset}]")
        return -1
    for offset in range(initial_offset, len(data) - (min_length * 2) + 1):
        is_potential_start = True
        for i in range(min_length):
            char_byte_offset = offset + i * 2
            if char_byte_offset + 1 >= len(data): is_potential_start = False; break
            low_byte = data[char_byte_offset]; high_byte = data[char_byte_offset + 1]
            is_valid_char = (32 <= low_byte <= 126 or low_byte in (0x09, 0x0a, 0x0d)) and high_byte == 0x00
            if not is_valid_char: is_potential_start = False; break
        if is_potential_start:
            if verbose: print(f"      [find_text_start: Found plausible start at offset {offset}]")
            return offset
    if verbose: print(f"      [find_text_start: No plausible start found from offset {initial_offset}]")
    return -1

def clean_segment_text(text):
    """Strips leading/trailing whitespace and trailing non-printable characters."""
    cleaned = text.strip();
    while cleaned and (ord(cleaned[-1]) < 32 or ord(cleaned[-1]) == 127):
        cleaned = cleaned[:-1]
    return cleaned

def extract_ascii_strings(data, min_length=4):
    """Extracts sequences of printable ASCII characters (and CR, LF, TAB) from byte data."""
    strings = []; current_string = ""
    for byte in data:
        if 32 <= byte <= 126 or byte in (ord('\n'), ord('\r'), ord('\t')):
            current_string += chr(byte)
        else:
            if len(current_string) >= min_length: strings.append(current_string)
            current_string = ""
    if len(current_string) >= min_length: strings.append(current_string)
    return "\n".join(strings)

# --- Unified Pattern Parsing Logic ---

# find_pattern_at takes verbose flag
def find_pattern_at(data_slice, index, verbose=False):
    """Checks for ADD/DELETE patterns. Includes verbose debug prints."""
    slice_len = len(data_slice)
    if verbose:
         try:
             debug_bytes_len = min(4, slice_len - index)
             debug_bytes = data_slice[index:index+debug_bytes_len]
             print(f"    [DEBUG find_pattern_at] Index={index:03d} (0x{index:02X}), Checking Bytes: {debug_bytes.hex(' ').upper()}")
         except Exception as e:
             print(f"    [DEBUG find_pattern_at] Index={index:03d} (0x{index:02X}), Error slicing: {e}")
    # Check ADD
    if index + 4 <= slice_len:
        pos, b1, b2, char = data_slice[index:index+4]
        is_add_match = (b1 == 0x00 and b2 == 0x01)
        if verbose: print(f"        ADD Check: Bytes={pos:02X} {b1:02X} {b2:02X} {char:02X} -> Match? {is_add_match}")
        if is_add_match:
             if verbose: print(f"        -> ADD MATCH FOUND!")
             return ('add', pos, char, 4)
    # Check DELETE
    if index + 3 <= slice_len:
        pos, b1, b2 = data_slice[index:index+3]
        is_del_match = (b1 == 0x01 and b2 == 0x00)
        if verbose: print(f"        DEL Check: Bytes={pos:02X} {b1:02X} {b2:02X} -> Match? {is_del_match}")
        if is_del_match:
             if verbose: print(f"        -> DEL MATCH FOUND!")
             return ('delete', pos, None, 3)
    if verbose: print(f"        -> NO MATCH.")
    return None

# UNIFIED Parser - Always Advance by 1
def parse_operations_PATTERN(data_slice, base_text_length=0, verbose=False): # Removed validate_sequential_adds
    """
    UNIFIED PARSER: Checks every byte offset, always advances index by 1.
    Records all structural matches found. Rule 4 validation is NOT performed.
    """
    found_ops_relative_temp = []
    current_index = 0
    slice_len = len(data_slice)
    if verbose: print("\n--- Parsing Operations (ALWAYS ADVANCE 1 Mode) ---"); sys.stdout.flush()

    while current_index < slice_len:
        # Pass verbose flag down
        result = find_pattern_at(data_slice, current_index, verbose=verbose)

        if result:
            # Record ANY structurally matching pattern found starting here
            op_type, pos, char_code, length = result
            found_ops_relative_temp.append({'index': current_index, 'type': op_type, 'pos': pos, 'char': char_code})
            if verbose: print(f"  Slice Idx={current_index:03d}: RECORDING potential pattern: Type={op_type}, Pos=0x{pos:02X}, Length={length}")

        # *** ALWAYS advance index by 1 to check the next offset ***
        current_index += 1

    if verbose: print("--- Pattern Scan Complete (ALWAYS ADVANCE 1 Mode) ---"); sys.stdout.flush()

    # Convert to the final tuple format, keeping ALL potentially overlapping patterns found
    final_ops_relative = [(op['index'], op['type'], op['pos'], op['char']) for op in found_ops_relative_temp]
    # Sort by index is crucial for application order
    final_ops_relative.sort(key=lambda x: x[0])

    # Rule 4 Validation is incompatible and Removed
    if verbose: print("\n--- Rule 4 Validation SKIPPED (Incompatible with ALWAYS ADVANCE 1 parser) ---")

    # Print the raw list found by this parser
    if verbose: print("\n--- Final Pattern Operations List (ALWAYS ADVANCE 1, Unfiltered) ---")
    if verbose:
        if not final_ops_relative: print("   (None)")
        else:
            for op in final_ops_relative: print(f"   ({op[0]:03d}, '{op[1]}', 0x{op[2]:02X}, {f'0x{op[3]:02X}' if op[3] is not None else 'None'})")
        sys.stdout.flush()

    return final_ops_relative

# UNIFIED Apply Edits Logic - Insert/Shift
def apply_edits_PATTERN(operations, initial_text="", verbose=False): # Renamed from INSERT
    """
    UNIFIED APPLY LOGIC: Applies ADD/DELETE operations assuming INSERT/SHIFT behavior (List based).
    """
    text_list = list(initial_text)
    if verbose: print("\n--- Applying Pattern Edits (INSERT/SHIFT Mode) ---"); print(f"    Starting with initial text (len={len(initial_text)}): {repr(initial_text)}"); sys.stdout.flush()
    if not operations: return initial_text

    # Operations should already be sorted by index from parser
    # operations.sort(key=lambda op: op[0]) # Ensure sorted if not already

    for index, op_type, position, char_code in operations:
        char = None
        # Decode char (same robust decoding as before)
        if op_type == 'add' and char_code is not None:
            try:
                char_bytes = bytes([char_code]); char = char_bytes.decode('utf-8', errors='replace')
                if not char.isprintable() and char not in ('\n', '\r', '\t', ' '): char = '.'
            except Exception: char = '.' if not (32 <= char_code <= 126) else chr(char_code)

        if verbose:
            current_len = len(text_list)
            print(f"Orig Idx {index:03d}: Op={op_type}, Pos=0x{position:02X}, Char='{repr(char)[1:-1] if char else 'N/A'}' ({f'0x{char_code:02X}' if char_code is not None else 'N/A'}), Current len={current_len}, List State: {repr(''.join(text_list))[:80]}")

        try:
            if op_type == 'add':
                # --- INSERT AT logic ---
                insert_pos = max(0, min(position, len(text_list)))
                if char is not None:
                     text_list.insert(insert_pos, char)
                     if verbose: print(f"      -> Inserted '{repr(char)[1:-1]}' AT index {insert_pos}. New len={len(text_list)}")
                elif verbose: print(f"      -> Add op with None char skipped.")
            elif op_type == 'delete':
                # --- Standard Delete Logic (shifts left) ---
                delete_pos = position
                if 0 <= delete_pos < len(text_list):
                    deleted_char = text_list.pop(delete_pos)
                    if verbose: print(f"      -> Deleted '{repr(deleted_char)}' from index {delete_pos}. New len={len(text_list)}")
                elif verbose: print(f"      -> Delete position {delete_pos} out of bounds (len={len(text_list)}). Skipped.")
        except Exception as e:
             if verbose: print(f"      -> ERROR applying op: {e}")
             print(f"{C.RED}      [Error applying operation {op_type} at index {index}: {e}]{C.ENDC}")

    if verbose: print("--- Applying Pattern Edits Complete (INSERT/SHIFT Mode) ---"); sys.stdout.flush()
    return "".join(text_list)

# Fallback function now just calls the unified functions
def extract_text_via_pattern_fallback(data_bytes, source_description="", verbose=False): # Removed validate_sequence
    """Applies the UNIFIED pattern parser and apply logic as a fallback mechanism."""
    if verbose: print(f"  [Pattern Fallback called for {source_description} - Using UNIFIED Logic]")
    if not data_bytes or len(data_bytes) < 2: return ""
    start_index = 2
    if start_index >= len(data_bytes): return ""
    data_to_parse = data_bytes[start_index:]
    original_offset = start_index
    # Call the UNIFIED parser (Always Advance 1, No Rule 4)
    operations_relative = parse_operations_PATTERN(data_to_parse, base_text_length=0, verbose=verbose) # Pass base_text_length=0
    operations = []
    for rel_idx, op_type, op_pos, op_char in operations_relative:
        original_idx = rel_idx + original_offset
        operations.append((original_idx, op_type, op_pos, op_char))
    operations.sort(key=lambda op: op[0])
    # Apply edits using the UNIFIED INSERT/SHIFT logic
    final_text = apply_edits_PATTERN(operations, initial_text="", verbose=verbose)
    return final_text

# --- File Processing Logic ---

# process_saved_file_vNext now uses UNIFIED logic
def process_saved_file_vNext(data, filepath, dump_hex, verbose=False):
    """Processes saved files using UNIFIED Logic: Path, Base Text, ALWAYS ADVANCE 1 Parser, INSERT/SHIFT Apply Logic."""
    print(f"{C.BOLD}[{os.path.basename(filepath)}] [Saved - Using Unified Logic]{C.ENDC}") # Indicate unified logic
    if dump_hex: print("\nHex Dump (first 256 bytes):"); print(hex_dump(data[:256])); print("...") if len(data)>256 else None

    # --- Extract File Path --- (Syntax corrected)
    file_path_start_offset = 5; max_path_scan_len = 512; scan_end_offset = min(file_path_start_offset + max_path_scan_len, len(data))
    file_path = "[Path Extraction Error]"; valid_path_bytes = bytearray()
    if file_path_start_offset < len(data) -1:
        try: # Indent 1
            idx = file_path_start_offset # Indent 2
            while idx < scan_end_offset - 1: # Indent 2
                low, high = data[idx:idx+2] # Indent 3
                if low == 0 and high == 0: break # Indent 3, semicolon removed
                if high != 0: # Indent 3
                    if verbose: print(f"   [Path Extractor: Non-zero high byte 0x{high:02X} at {idx+1}]") # Indent 4
                    break # Indent 4, semicolon removed
                valid_path_bytes.extend([low, high]); idx += 2 # Indent 3, semicolon removed
            # AFTER WHILE LOOP
            if not valid_path_bytes: file_path = "[Path Extraction Failed]" # Indent 2
            else: file_path = decode_utf16le(valid_path_bytes) # Indent 2
        except Exception as e: # Indent 1 (aligned with try)
             file_path = f"[Path Extraction Error: {e}]" # Indent 2 (inside except)
             if verbose: print(f"   [Path Extractor: Exception during extraction: {e}]\n{traceback.format_exc(limit=1)}")

        print(f"\n{C.BLUE}File Path:{C.ENDC} {file_path}") # Indent 1 (after try-except)
    else: # Indent 1 (aligned with if)
        print(f"\n{C.YELLOW}[Warning: Not enough data for path]{C.ENDC}")
    print(C.BLUE + "-" * 40 + C.ENDC) # Indent 0
    # --- End Extract File Path ---

    # --- Extract Base Text --- (Unchanged)
    if verbose: print("--- Searching for Initial Content Block (02 01 01...) ---"); sys.stdout.flush()
    base_text = ""; end_base_text_idx = -1; base_text_marker = b'\x02\x01\x01'; base_text_found = False
    try:
        marker_idx = data.find(base_text_marker)
        if marker_idx != -1:
            len_byte_idx = marker_idx + len(base_text_marker)
            if len_byte_idx < len(data):
                length = data[len_byte_idx]; start_text_idx = len_byte_idx + 1; end_text_idx_calc = start_text_idx + length * 2
                if end_text_idx_calc <= len(data):
                    base_text = decode_utf16le(data[start_text_idx : end_text_idx_calc])
                    end_base_text_idx = end_text_idx_calc; base_text_found = True
                    if verbose: print(f"      [Info: Base Text Found: len={length}, data@[{start_text_idx}:{end_text_idx_calc}], raw='{repr(base_text)}']")
        if not base_text_found:
             if verbose: print("      [Info: Base text marker not found or data incomplete]")
    except Exception as e:
        if verbose: print(f"{C.RED}      [Error extracting base text: {e}]{C.ENDC}")
    base_text_len = len(base_text)
    if base_text_found: print(f"\n{C.GREEN}Initial Base Text ({base_text_len} chars):{C.ENDC}\n{base_text.replace('\r', '\n')}")
    else: print(f"{C.YELLOW}\n[No Initial Base Text Found]{C.ENDC}"); end_base_text_idx = 2 if end_base_text_idx == -1 else end_base_text_idx
    print(C.BLUE + "-" * 40 + C.ENDC)

    # --- Parse Operations using UNIFIED parser ---
    if verbose: print(f"--- Parsing ADD/DELETE Operations (ALWAYS ADVANCE 1 Mode, Rule 4 Disabled) ---"); print(f"      Scanning data from index {end_base_text_idx} onwards..."); sys.stdout.flush()
    final_reconstructed_text = "[Error]"; error_occurred = False
    try:
        data_to_parse_ops = data[end_base_text_idx:] if end_base_text_idx < len(data) else b''
        original_offset = end_base_text_idx
        if not data_to_parse_ops:
            if verbose: print("      [Info: No data after base text. Using base text.]")
            final_reconstructed_text = base_text
        else:
            # *** Call the UNIFIED parser (Always Advance 1) ***
            # Pass base_text_len only for info message, Rule 4 is disabled in parser
            operations_relative = parse_operations_PATTERN(data_to_parse_ops, base_text_length=base_text_len, verbose=verbose)
            operations = [(rel_idx + original_offset, op_type, op_pos, op_char) for rel_idx, op_type, op_pos, op_char in operations_relative]
            operations.sort(key=lambda op: op[0])
            # *** Call UNIFIED INSERT/SHIFT apply logic ***
            final_reconstructed_text = apply_edits_PATTERN(operations, initial_text=base_text, verbose=verbose)
    except Exception as pattern_error:
        print(f"\n{C.RED}[Error during saved op parsing/application: {pattern_error}]\n{traceback.format_exc()}{C.ENDC}")
        final_reconstructed_text = f"[Error: {pattern_error}]"; error_occurred = True

    # --- Final Print --- (Unchanged)
    if not error_occurred: print(f"\n{C.GREEN}--- Final Reconstructed Content ---{C.ENDC}"); print(f"{final_reconstructed_text.replace('\r', '\n')}")
    else: print(f"\n{C.RED}{final_reconstructed_text}{C.ENDC}")
    print(C.BLUE + "-" * 40 + C.ENDC)

# process_unsaved_file uses UNIFIED logic via fallback function
def process_unsaved_file(data, filepath, dump_hex, verbose=False):
    """Processes unsaved files using UNIFIED Logic: Primary UTF-16LE -> ASCII fallback -> ALWAYS ADVANCE 1 Parser + INSERT/SHIFT Apply Logic."""
    print(f"{C.BOLD}[{os.path.basename(filepath)}] [Unsaved]{C.ENDC}")
    if dump_hex: print("\nHex Dump (first 256 bytes):"); print(hex_dump(data[:256])); print("...") if len(data)>256 else None
    primary_extracted_text = ""; ascii_fallback_text = ""; pattern_fallback_text = "";
    primary_method_success = False
    # --- Primary Method --- (Syntax corrected)
    if verbose: print("--- Attempting Primary Method (UTF-16LE Heuristics) ---")
    text_start_offset = find_text_start(data, 0, verbose=verbose)
    separator_offset = -1 # Initialize first
    try: # Start try block on a NEW line
        separator_offset = data.rfind(b'\x01')
    except Exception as e: # Catch potential errors during rfind
        if verbose: print(f"      [Warning: data.rfind(b'\\x01') failed: {e}]")
        pass # Ignore if rfind fails
    if separator_offset != -1 and text_start_offset != -1 and text_start_offset < separator_offset:
        text_data = data[text_start_offset:separator_offset]
        if verbose: print(f"      [Primary Method: Found start={text_start_offset}, separator={separator_offset}. Decoding {len(text_data)} bytes.]")
        decoded_text = decode_utf16le(text_data); cleaned_text = clean_segment_text(decoded_text)
        if cleaned_text:
            primary_method_success = True; primary_extracted_text = cleaned_text
            print(f"\n{C.GREEN}File Content (UTF-16LE Section - Primary Method):{C.ENDC}"); print(f"{cleaned_text.replace('\r', '\n')}")
        else: print(f"\n{C.YELLOW}[Primary Method: UTF-16LE decoding yielded empty result. Trying fallbacks.]{C.ENDC}");
    else: # Print warnings if primary skipped (Unchanged)
        if verbose: print(f"      [Primary Method: Heuristics failed. Trying fallbacks.]")
        if separator_offset == -1 and text_start_offset == -1: print(f"\n{C.YELLOW}[Warning: Text start and 0x01 separator not found. Primary UTF-16LE skipped.]{C.ENDC}")
        elif separator_offset == -1: print(f"\n{C.YELLOW}[Warning: 0x01 separator not found. Primary UTF-16LE skipped.]{C.ENDC}")
        elif text_start_offset == -1: print(f"\n{C.YELLOW}[Warning: UTF-16LE text start not found. Primary UTF-16LE skipped.]{C.ENDC}")
        elif text_start_offset >= separator_offset: print(f"\n{C.YELLOW}[Warning: Text start ({text_start_offset}) >= separator ({separator_offset}). Primary UTF-16LE skipped.]{C.ENDC}")
    # --- ASCII Fallback --- (Unchanged)
    if not primary_method_success:
        if verbose: print(f"\n{C.YELLOW}--- Applying ASCII Fallback to Full File Data ---{C.ENDC}")
        try:
            ascii_fallback_text = extract_ascii_strings(data)
            if ascii_fallback_text: print(f"\n{C.YELLOW}Fallback Extracted File Content (ASCII Strings):{C.ENDC}"); print(ascii_fallback_text.replace('\r', '\n'))
            elif verbose: print(f"{C.YELLOW}   [No Content Extracted (ASCII Fallback)]{C.ENDC}")
        except Exception as ascii_error: print(f"\n{C.RED}[ASCII Fallback Error: {ascii_error}]{C.ENDC}"); print(traceback.format_exc(limit=1)) if verbose else None
    # --- Use UNIFIED Fallback ---
    if verbose: print(f"\n{C.BLUE}--- Applying PATTERN PARSER Fallback (UNIFIED Logic) ---{C.ENDC}")
    sys.stdout.flush()
    pattern_error_occurred = False
    try:
        # *** Call the fallback function using UNIFIED logic ***
        pattern_fallback_text = extract_text_via_pattern_fallback(data, filepath, verbose=verbose) # validate_sequence removed
    except Exception as pattern_error:
        print(f"\n{C.RED}[Pattern Parser Fallback Error: {pattern_error}]\nTraceback:\n{traceback.format_exc()}{C.ENDC}"); pattern_error_occurred = True; pattern_fallback_text = f"[Error: {pattern_error}]"
    if not pattern_error_occurred:
        if pattern_fallback_text:
            should_print_pattern = True # Check for duplicates vs other methods
            if primary_extracted_text and primary_extracted_text == pattern_fallback_text: should_print_pattern = False;
            elif ascii_fallback_text and ascii_fallback_text.replace('\r', '\n') == pattern_fallback_text.replace('\r', '\n'): should_print_pattern = False;
            if should_print_pattern:
                print(f"\n{C.YELLOW}Extracted File Content (PATTERN PARSER Fallback - UNIFIED Logic):{C.ENDC}") # Indicate mode
                if verbose: print(f"Raw pattern text: {repr(pattern_fallback_text)}")
                print(f"{pattern_fallback_text.replace('\r', '\n')}") # Apply final format
            elif verbose: print(f"{C.YELLOW}[PATTERN PARSER Fallback result matches previous output, skipping duplicate print]{C.ENDC}")
        elif verbose: print(f"{C.YELLOW}   [No Content Extracted (PATTERN PARSER Fallback - UNIFIED Logic)]{C.ENDC}") # Indicate mode
    print(C.BLUE + "-" * 40 + C.ENDC)

# process_file_vNext passes verbose down
def process_file_vNext(filepath, dump_hex, verbose=False):
    """Reads a single file, determines type (saved/unsaved), and calls the appropriate processor."""
    print(f"\n{C.HEADER}>>> Processing file: {os.path.basename(filepath)}{C.ENDC}")
    try:
        with open(filepath, 'rb') as f: data = f.read()
        if len(data) < 4: print(f"[{os.path.basename(filepath)}] [Unknown/Too Short]"); print(f"{C.RED}[Error] File is too short (less than 4 bytes){C.ENDC}"); print("-" * 40); return
        is_saved = (data[3] == 1)
        # Pass verbose flag to the specific processors
        if is_saved: process_saved_file_vNext(data, filepath, dump_hex, verbose=verbose)
        else: process_unsaved_file(data, filepath, dump_hex, verbose=verbose)
    except FileNotFoundError: print(f"{C.RED}[Error: File not found: {filepath}]{C.ENDC}"); print("-" * 40)
    except PermissionError: print(f"{C.RED}[Error: Permission denied: {filepath}]{C.ENDC}"); print("-" * 40)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info(); fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"{C.RED}[Error: Unexpected error processing {os.path.basename(filepath)}: {e} in {fname} at line {exc_tb.tb_lineno}]{C.ENDC}"); print(traceback.format_exc()); print("-" * 40)

# process_directory_vNext passes verbose down
def process_directory_vNext(directory_path, dump_hex, verbose=False):
    """Processes all relevant .bin files in a given directory."""
    if not os.path.isdir(directory_path): print(f"{C.RED}Error: Directory not found: {directory_path}{C.ENDC}"); return
    all_files = sorted(glob.glob(os.path.join(directory_path, "*.bin")))
    files_to_process = [f for f in all_files if not (f.endswith(".0.bin") or f.endswith(".1.bin"))]
    processed_count = 0; print(f"Found {len(all_files)} total .bin files, processing {len(files_to_process)} relevant files.")
    for filepath in files_to_process:
        process_file_vNext(filepath, dump_hex, verbose=verbose) # Pass verbose down
        processed_count += 1
    if processed_count == 0: print(f"\n{C.YELLOW}No suitable .bin files found or processed in {directory_path}{C.ENDC}")
    else: print(f"\n{C.GREEN}Finished processing {processed_count} file(s).{C.ENDC}")

# get_notepad_tabstate_path (Unchanged)
def get_notepad_tabstate_path():
    """Attempts to find the default Notepad TabState directory."""
    localappdata = os.environ.get('LOCALAPPDATA')
    if localappdata:
        notepad_path = os.path.join(localappdata, r"Packages\Microsoft.WindowsNotepad_8wekyb3d8bbwe\LocalState\TabState")
        if os.path.isdir(notepad_path): return notepad_path
    return None

# --- Main Execution Block ---
if __name__ == '__main__':
    # Timezone context (Using current time at execution)
    user_tz_name = 'Europe/London'; context_dt_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " (System TZ Unknown)"
    try:
        if pytz: user_tz = pytz.timezone(user_tz_name); context_dt = datetime.datetime.now(user_tz); context_dt_str = context_dt.strftime('%Y-%m-%d %H:%M:%S %Z%z') + f" ({user_tz_name})"
        else: raise Exception("pytz not available")
    except Exception as e:
        print(f"{C.YELLOW}Warning: Could not use specified timezone '{user_tz_name}' ({e}). Using system default timezone.{C.ENDC}", file=sys.stderr)
        try: context_dt = datetime.datetime.now().astimezone(); tz_name = context_dt.tzname(); offset_str = context_dt.strftime('%z'); tz_display = f" {tz_name}" if tz_name else ""; context_dt_str = context_dt.strftime('%Y-%m-%d %H:%M:%S') + f"{tz_display}{offset_str}"
        except Exception: pass
    print(f"{C.HEADER}--- Notepad Decoder --- Running at {context_dt_str} ---{C.ENDC}"); print("")

    # Arg Parsing (Unchanged)
    parser = argparse.ArgumentParser(description="Decode Notepad .bin files (TabState) using unified experimental logic.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-f", "--file", help="Path to a single .bin file to process.")
    parser.add_argument("-d", "--directory", help="Path to the directory containing .bin files (e.g., TabState).\nDefault: Auto-detect standard location.")
    parser.add_argument("--no-hex", action="store_true", help="Disable partial hex dump output.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug output.")
    args = parser.parse_args()

    # Set processing flags from args
    verbose_setting = args.verbose
    dump_hex_enabled = not args.no_hex

    # Execute
    if args.file:
        if os.path.isfile(args.file): print(f"Processing single file: {args.file}"); process_file_vNext(args.file, dump_hex_enabled, verbose=verbose_setting)
        else: print(f"{C.RED}Error: Specified file not found: {args.file}{C.ENDC}", file=sys.stderr); sys.exit(1)
    else:
        target_directory = args.directory
        if not target_directory: print("No directory specified via -d, attempting auto-detection..."); target_directory = get_notepad_tabstate_path()
        if target_directory:
            print(f"Using directory: {target_directory}")
            if not os.path.isdir(target_directory): print(f"{C.RED}Error: Target directory not found: {target_directory}{C.ENDC}", file=sys.stderr); parser.print_help(file=sys.stderr); sys.exit(1)
            process_directory_vNext(target_directory, dump_hex_enabled, verbose=verbose_setting)
        else: print(f"\n{C.RED}Error: Could not automatically find TabState directory and none was specified via -d.{C.ENDC}", file=sys.stderr); parser.print_help(file=sys.stderr); sys.exit(1)
