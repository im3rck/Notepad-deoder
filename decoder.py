import os
import argparse
import glob
import collections
import datetime
import sys

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

# --- Helper Functions ---

def hex_dump(data, bytes_per_line=16):
    # (Function remains unchanged)
    output = []; hex_part = ''; text_part = ''
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]; hex_part = ' '.join(f'{b:02X}' for b in chunk)
        text_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        output.append(f'{i:08X}: {hex_part.ljust(bytes_per_line * 3)}  {text_part}')
    return '\n'.join(output)

def decode_utf16le(data):
    # (Function remains unchanged)
    try:
        text = data.decode('utf-16le', errors='ignore')
        return text.rstrip('\x00')
    except Exception as e:
        return f"[UTF-16LE Decode Error: {e}]"

def find_text_start(data, initial_offset=0, min_length=2, verbose=False):
    # (Function remains unchanged)
    for offset in range(initial_offset, len(data) - (min_length * 2) + 1):
        is_potential_start = True
        for i in range(min_length):
            char_byte_offset = offset + i * 2
            if char_byte_offset + 1 >= len(data): is_potential_start = False; break
            low_byte = data[char_byte_offset]; high_byte = data[char_byte_offset + 1]
            is_valid_char = (32 <= low_byte <= 126 or low_byte in (0x09, 0x0a, 0x0d)) and high_byte == 0x00 # Allow CR, LF, TAB
            if not is_valid_char: is_potential_start = False; break
        if is_potential_start:
            if verbose: print(f"    [find_text_start: Found plausible start at offset {offset}]")
            return offset
    if verbose: print(f"    [find_text_start: No plausible start found from offset {initial_offset}]")
    return -1

def clean_segment_text(text):
    # (Function remains unchanged)
    cleaned = text.strip();
    while cleaned and (ord(cleaned[-1]) < 32 or ord(cleaned[-1]) == 127): cleaned = cleaned[:-1]
    return cleaned

def extract_ascii_strings(data, min_length=4):
    # (Function remains unchanged)
    strings = []; current_string = ""
    for byte in data:
        if 32 <= byte <= 126 or byte in (ord('\n'), ord('\r'), ord('\t')): current_string += chr(byte)
        else:
            if len(current_string) >= min_length: strings.append(current_string)
            current_string = ""
    if len(current_string) >= min_length: strings.append(current_string)
    return "\n".join(strings)

# --- Pattern Parsing Logic (Used as Fallback) ---

def find_pattern_at(data_slice, index):
    # (Function remains unchanged)
    slice_len = len(data_slice)
    if index + 4 <= slice_len: # Check ADD
        pos, b1, b2, char = data_slice[index:index+4]
        if b1 == 0x00 and b2 == 0x01: return ('add', pos, char, 4)
    if index + 3 <= slice_len: # Check DELETE
        pos, b1, b2 = data_slice[index:index+3]
        if b1 == 0x01 and b2 == 0x00: return ('delete', pos, None, 3)
    return None

def parse_operations_PATTERN(data_slice, validate_sequential_adds=False, base_text_length=0, verbose=False):
    # (Function remains unchanged - includes bug fix)
    found_ops_relative_temp = []
    current_index = 0; slice_len = len(data_slice)
    final_ops_relative = []
    if verbose: print("\n--- Parsing Operations (Pattern Fallback) ---"); sys.stdout.flush()
    while current_index < slice_len:
        result = find_pattern_at(data_slice, current_index)
        if result:
            op_type, pos, char_code, length = result
            found_ops_relative_temp.append({'index': current_index, 'type': op_type, 'pos': pos, 'char': char_code})
            if verbose: print(f"  Slice Idx={current_index:03d}: FOUND Pattern: Type={op_type}, Pos=0x{pos:02X}, Char={f'0x{char_code:02X}' if char_code is not None else 'N/A'}, Length={length}")
            current_index += length
        else:
            current_index += 1
    if verbose: print("--- Pattern Scan Complete ---"); sys.stdout.flush()
    if validate_sequential_adds:
        adds_for_validation = sorted([op for op in found_ops_relative_temp if op['type'] == 'add'], key=lambda x: (x['pos'], x['index']))
        valid_add_indices = set(); last_valid_sequential_pos = base_text_length - 1
        if verbose: print(f"\n--- Applying Rule 4 Validation (Seq Starts After Base Len {base_text_length}, Exc. Enabled) ---"); print(f"    Initial expected sequential pos = {last_valid_sequential_pos + 1}"); sys.stdout.flush()
        if not adds_for_validation and verbose: print("  (No ADD operations found to validate)")
        elif adds_for_validation:
            deletes_by_pos_and_index = collections.defaultdict(list)
            for op in found_ops_relative_temp:
                if op['type'] == 'delete': deletes_by_pos_and_index[op['pos']].append(op['index'])
            for pos_key in deletes_by_pos_and_index: deletes_by_pos_and_index[pos_key].sort()
            for add_op in adds_for_validation:
                op_index = add_op['index']; pos = add_op['pos']; is_valid_current_add = False; exception_applied = False
                is_sequential = (pos == last_valid_sequential_pos + 1)
                if is_sequential: is_valid_current_add = True
                else:
                    is_within_base = (pos < base_text_length)
                    if is_within_base:
                        preceding_delete_exists = any(del_idx < op_index for del_idx in deletes_by_pos_and_index.get(pos, []))
                        if preceding_delete_exists: is_valid_current_add = True; exception_applied = True
                if is_valid_current_add:
                     if verbose: print(f"  -> Valid ADD: RelIdx={op_index:03d}, Pos=0x{pos:02X}. (Seq OK {'w/ Exc.' if exception_applied else ''})")
                     valid_add_indices.add(op_index)
                     if is_sequential and pos > last_valid_sequential_pos: last_valid_sequential_pos = pos
                elif verbose: print(f"  -> INVALID ADD: RelIdx={op_index:03d}, Pos=0x{pos:02X}. Expected seq pos {last_valid_sequential_pos + 1}, no exception applied. IGNORING.")
            if verbose: sys.stdout.flush()
        validated_ops_unsorted = []
        for op in sorted(found_ops_relative_temp, key=lambda x: x['index']):
            if op['type'] == 'delete': validated_ops_unsorted.append((op['index'], op['type'], op['pos'], op['char']))
            elif op['type'] == 'add' and op['index'] in valid_add_indices: validated_ops_unsorted.append((op['index'], op['type'], op['pos'], op['char']))
        final_ops_relative = sorted(validated_ops_unsorted, key=lambda x: x[0])
        if verbose: print("\n--- Validated Pattern Operations List (Relative Indices) ---")
    else:
        if verbose: print("\n--- Rule 4 Validation SKIPPED for Pattern Parser ---"); sys.stdout.flush()
        final_ops_relative = sorted([(op['index'], op['type'], op['pos'], op['char']) for op in found_ops_relative_temp ], key=lambda x: x[0])
        if verbose: print("\n--- Final Pattern Operations List (Relative Indices, Rule 4 Skipped) ---")
    if verbose:
        if not final_ops_relative: print("  (None)")
        else:
            for op in final_ops_relative: print(f"  ({op[0]:03d}, '{op[1]}', {op[2]}, {op[3]})")
        sys.stdout.flush()
    return final_ops_relative

def apply_edits_PATTERN(operations, initial_text="", verbose=False):
    # (Function remains unchanged - includes IndentationError fix)
    text_state = {i: char for i, char in enumerate(initial_text)}
    position_states = {i: 'ADDED' for i in range(len(initial_text))}
    if verbose: print("\n--- Applying Pattern Edits ---"); print(f"    Starting with initial text (len={len(initial_text)}): {repr(initial_text)}"); sys.stdout.flush()
    if not operations:
        if verbose: print("  (No edit operations to apply)")
        return initial_text
    for index, op_type, position, char_code in operations:
        char = None
        if op_type == 'add' and char_code is not None:
            try:
                char_bytes = bytes([char_code])
                try: char = char_bytes.decode('utf-8')
                except UnicodeDecodeError: char = char_bytes.decode('latin-1', errors='replace')
                if not char.isprintable() and char not in ('\n', '\r', '\t', ' '): char = '.'
            except Exception:
                 if 32 <= char_code <= 126: char = chr(char_code)
                 else: char = '.'
        current_state = position_states.get(position)
        if verbose: print(f"Orig Idx {index:03d}: Applying Op: {op_type}, Pos: 0x{position:02X}, Char: '{repr(char)[1:-1] if char else 'N/A'}' ({f'0x{char_code:02X}' if char_code is not None else 'N/A'})")
        if op_type == 'add':
            text_state[position] = char; position_states[position] = 'ADDED'
        elif op_type == 'delete':
            if current_state == 'ADDED': position_states[position] = 'DELETED'
    if verbose: print("--- Applying Pattern Edits Complete ---"); sys.stdout.flush()
    sorted_positions = sorted(text_state.keys())
    final_text = "".join([text_state[pos] for pos in sorted_positions if position_states.get(pos) == 'ADDED'])
    return final_text


def extract_text_via_pattern_fallback(data_bytes, source_description="", validate_sequence=False, verbose=False):
    # (Function remains unchanged)
    if verbose: print(f"  [Pattern Fallback called for {source_description}, validate_sequence={validate_sequence}]")
    if not data_bytes or len(data_bytes) < 2: return ""
    start_index = 2 # Rule 2
    if start_index >= len(data_bytes): return ""
    data_to_parse = data_bytes[start_index:]
    original_offset = start_index
    operations_relative = parse_operations_PATTERN(data_to_parse, validate_sequential_adds=validate_sequence, base_text_length=0, verbose=verbose)
    operations = []
    for rel_idx, op_type, op_pos, op_char in operations_relative:
        original_idx = rel_idx + original_offset
        operations.append((original_idx, op_type, op_pos, op_char))
    operations.sort(key=lambda op: op[0])
    final_text = apply_edits_PATTERN(operations, initial_text="", verbose=verbose)
    return final_text


# --- File Processing Logic ---

def process_saved_file_vNext(data, filepath, dump_hex, verbose=False):
    """Processes saved files: Path, Base Text, Pattern Parser (Rule 4 ON). Applies \r -> \n formatting."""
    print(f"{C.BOLD}[{os.path.basename(filepath)}] [Saved]{C.ENDC}")
    if dump_hex: print("\nHex Dump (first 256 bytes):"); print(hex_dump(data[:256])); print("...") if len(data)>256 else None
    file_path_start_offset = 5
    max_path_scan_len = 256
    scan_end_offset = min(file_path_start_offset + max_path_scan_len, len(data))
    file_path = "[Path Extraction Error]"; valid_path_bytes = bytearray()
    if file_path_start_offset < len(data):
        try:
            idx = file_path_start_offset
            while idx < scan_end_offset - 1:
                low_byte = data[idx]; high_byte = data[idx+1];
                if high_byte != 0x00: break;
                if low_byte == 0x00 and high_byte == 0x00: break
                valid_path_bytes.append(low_byte); valid_path_bytes.append(high_byte); idx += 2
            if not valid_path_bytes: file_path = "[Path Extraction Failed]"
            else: file_path = decode_utf16le(valid_path_bytes)
        except Exception as path_extract_err: file_path = f"[Path Extraction Error: {path_extract_err}]"
        print(f"\n{C.BLUE}File Path:{C.ENDC} {file_path}")
    else: print(f"\n{C.YELLOW}[Warning: Cannot extract file path]{C.ENDC}")
    print(C.BLUE + "-" * 40 + C.ENDC)

    # --- Find and Extract Initial Base Content Block ---
    if verbose: print("--- Searching for Initial Content Block (02 01 01...) ---"); sys.stdout.flush()
    base_text = ""; end_base_text_idx = 2; base_text_marker = b'\x02\x01\x01'
    base_text_found = False
    try:
        marker_idx = data.find(base_text_marker)
        if marker_idx != -1:
            len_idx = marker_idx + len(base_text_marker)
            if len_idx < len(data):
                length = data[len_idx]; start_text_idx = len_idx + 1; end_text_idx_calc = start_text_idx + length * 2
                if end_text_idx_calc <= len(data):
                    base_text_bytes = data[start_text_idx : end_text_idx_calc]
                    base_text = decode_utf16le(base_text_bytes) # Keeps \r
                    end_base_text_idx = end_text_idx_calc; base_text_found = True
                    if verbose: print(f"    [Info: Found Base Text Marker at {marker_idx}, Length={length}. Base text ends at {end_base_text_idx}]")
                    if verbose: print(f"    Raw base text: {repr(base_text)}")
                elif verbose: print(f"    [Warning: Base text end index ({end_text_idx_calc}) > data length ({len(data)}).]")
            elif verbose: print(f"    [Warning: Length byte index ({len_idx}) > data length ({len(data)}).]")
        elif verbose: print("    [Info: Initial content marker not found.]")
    except Exception as base_extract_err: import traceback; print(f"{C.RED}    [Error extracting base text: {base_extract_err}]\n{traceback.format_exc()}{C.ENDC}")
    base_text_len = len(base_text)
    if base_text_found: print(f"\n{C.GREEN}Initial Base Text ({base_text_len} chars):{C.ENDC}\n{base_text.replace('\r', '\n')}") # Apply formatting here too
    else: print(f"{C.YELLOW}\n[No Initial Base Text Found]{C.ENDC}")
    print(C.BLUE + "-" * 40 + C.ENDC)

    # --- Parse Subsequent Operations and Apply to Base Text ---
    if verbose: print("--- Parsing ADD/DELETE Operations (Rule 4 Enabled + Exc.) ---"); print(f"    Scanning data from index {end_base_text_idx} onwards..."); sys.stdout.flush()
    final_reconstructed_text = "[Error during operation parsing/application]"; error_occurred = False
    try:
        data_to_parse_ops = data[end_base_text_idx:] if end_base_text_idx < len(data) else b''
        original_offset = end_base_text_idx
        if not data_to_parse_ops:
             if verbose: print("    [Info: No data found after base text block.]")
             final_reconstructed_text = base_text # Use original base text
        else:
             operations_relative = parse_operations_PATTERN(data_to_parse_ops, validate_sequential_adds=True, base_text_length=base_text_len, verbose=verbose)
             operations = [];
             for rel_idx, op_type, op_pos, op_char in operations_relative: operations.append((rel_idx + original_offset, op_type, op_pos, op_char))
             operations.sort(key=lambda op: op[0])
             final_reconstructed_text = apply_edits_PATTERN(operations, initial_text=base_text, verbose=verbose)
    except Exception as pattern_error: import traceback; print(f"\n{C.RED}[Error during operation parsing/application: {pattern_error}]\n{traceback.format_exc()}{C.ENDC}"); final_reconstructed_text = "[Error]"; error_occurred = True

    # --- Final Print for Saved Files ---
    if not error_occurred:
        print(f"\n{C.GREEN}--- Final Reconstructed Content ---{C.ENDC}")
        if verbose: print(f"Raw final text: {repr(final_reconstructed_text)}")
        # --- APPLY FORMATTING ---
        formatted_final_text = final_reconstructed_text.replace('\r', '\n')
        print(f"{formatted_final_text}")
        # ----------------------
    else: print(f"\n{C.RED}{final_reconstructed_text}{C.ENDC}")
    print(C.BLUE + "-" * 40 + C.ENDC)


# --- UPDATED process_unsaved_file ---
def process_unsaved_file(data, filepath, dump_hex, verbose=False):
    """Processes unsaved files: Primary UTF-16LE -> ASCII fallback -> Pattern Parser fallback (Rule 4 OFF). Applies \r -> \n formatting."""
    print(f"{C.BOLD}[{os.path.basename(filepath)}] [Unsaved]{C.ENDC}")
    if dump_hex: print("\nHex Dump (first 256 bytes):"); print(hex_dump(data[:256])); print("...") if len(data) > 256 else None
    primary_extracted_text = ""; ascii_fallback_text = ""; pattern_fallback_text = ""; # Store originals
    primary_method_success = False
    separator_offset = data.rfind(b'\x01'); text_start_offset = find_text_start(data, 0, verbose=verbose)

    # --- Primary Method ---
    if separator_offset != -1 and text_start_offset != -1 and text_start_offset < separator_offset:
        primary_method_success = True
        text_data = data[text_start_offset:separator_offset]
        decoded_text = decode_utf16le(text_data) # Keeps \r
        cleaned_text = clean_segment_text(decoded_text)
        if cleaned_text:
            print(f"\n{C.GREEN}File Content (UTF-16LE Section - Primary Method):{C.ENDC}")
            if verbose: print(f"Raw primary text: {repr(cleaned_text)}")
            # --- APPLY FORMATTING ---
            formatted_text = cleaned_text.replace('\r', '\n')
            print(f"{formatted_text}")
            # ----------------------
            primary_extracted_text = cleaned_text # Store original
        else:
            print(f"\n{C.YELLOW}[Primary Method: UTF-16LE decoding yielded empty result.]{C.ENDC}"); primary_method_success = False
    else: # Print warnings if primary skipped
        if separator_offset == -1: print(f"\n{C.YELLOW}[Warning: 0x01 separator not found. Primary UTF-16LE skipped.]{C.ENDC}")
        elif text_start_offset == -1: print(f"\n{C.YELLOW}[Warning: UTF-16LE text start not found. Primary UTF-16LE skipped.]{C.ENDC}")
        elif text_start_offset >= separator_offset: print(f"\n{C.YELLOW}[Warning: Text start ({text_start_offset}) >= separator ({separator_offset}). Primary UTF-16LE skipped.]{C.ENDC}")

    # --- ASCII Fallback ---
    if not primary_method_success:
        if verbose: print(f"\n{C.YELLOW}--- Applying ASCII Fallback to Full File Data ---{C.ENDC}")
        try:
            ascii_fallback_text = extract_ascii_strings(data)
            if ascii_fallback_text: # Only print if something was extracted
                 print(f"\n{C.YELLOW}Fallback Extracted File Content (ASCII Strings):{C.ENDC}")
                 if verbose: print(f"Raw ASCII text: {repr(ascii_fallback_text)}")
                 # --- APPLY FORMATTING ---
                 formatted_text = ascii_fallback_text.replace('\r', '\n')
                 print(formatted_text)
                 # ----------------------
            elif verbose: print(f"{C.YELLOW}[No Content Extracted (ASCII Fallback)]{C.ENDC}")
        except Exception as ascii_error: print(f"\n{C.RED}[ASCII Fallback Error: {ascii_error}]{C.ENDC}")

    # --- ALWAYS Run PATTERN PARSER Fallback (Rule 4 DISABLED for Unsaved) ---
    if verbose: print(f"\n{C.BLUE}--- Applying PATTERN PARSER Fallback (Rule 4 Disabled) ---{C.ENDC}")
    sys.stdout.flush()
    pattern_error_occurred = False
    try:
        pattern_fallback_text = extract_text_via_pattern_fallback(data, filepath, validate_sequence=False, verbose=verbose)
    except Exception as pattern_error: import traceback; tb_str = traceback.format_exc(); print(f"\n{C.RED}[Pattern Parser Fallback Error: {pattern_error}]\nTraceback:\n{tb_str}{C.ENDC}"); pattern_error_occurred = True

    if not pattern_error_occurred:
        if pattern_fallback_text:
            # Avoid printing duplicate info if primary/ASCII already succeeded AND produced same content
            should_print_pattern = True
            if primary_extracted_text and primary_extracted_text == pattern_fallback_text: should_print_pattern = False; # Already shown via primary
            if ascii_fallback_text and ascii_fallback_text == pattern_fallback_text: should_print_pattern = False; # Already shown via ASCII

            if should_print_pattern:
                 print(f"\n{C.YELLOW}Extracted File Content (PATTERN PARSER Fallback):{C.ENDC}")
                 if verbose: print(f"Raw pattern text: {repr(pattern_fallback_text)}")
                 # --- APPLY FORMATTING ---
                 formatted_text = pattern_fallback_text.replace('\r', '\n')
                 print(f"{formatted_text}")
                 # ----------------------
            elif verbose: print(f"{C.YELLOW}[PATTERN PARSER Fallback result matches previous output, skipping duplicate print]{C.ENDC}")
        elif verbose: print(f"{C.YELLOW}[No Content Extracted (PATTERN PARSER Fallback)]{C.ENDC}")
    print(C.BLUE + "-" * 40 + C.ENDC)
# --- END UPDATED process_unsaved_file ---


# --- File/Directory Handling & Main Execution ---
def process_file_vNext(filepath, dump_hex, verbose=False):
    # (Function remains unchanged)
    print(f"\n{C.HEADER}>>> Processing file: {os.path.basename(filepath)}{C.ENDC}")
    try:
        with open(filepath, 'rb') as f: data = f.read()
        if len(data) < 4: print(f"[{os.path.basename(filepath)}] [Unknown/Too Short]"); print(f"{C.RED}[Error] File is too short{C.ENDC}"); print("-" * 40); return
        is_saved = data[3] == 1
        if is_saved: process_saved_file_vNext(data, filepath, dump_hex, verbose=verbose)
        else: process_unsaved_file(data, filepath, dump_hex, verbose=verbose)
    except FileNotFoundError: print(f"{C.RED}[Error: File not found: {filepath}]{C.ENDC}"); print("-" * 40)
    except PermissionError: print(f"{C.RED}[Error: Permission denied: {filepath}]{C.ENDC}"); print("-" * 40)
    except Exception as e:
        import traceback; exc_type, exc_obj, exc_tb = sys.exc_info(); fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"{C.RED}[Error: Unexpected error processing {os.path.basename(filepath)}: {e} in {fname} at line {exc_tb.tb_lineno}]{C.ENDC}"); print(traceback.format_exc()); print("-" * 40)

def process_directory_vNext(directory_path, dump_hex, verbose=False):
    # (Function remains unchanged)
    if not os.path.isdir(directory_path): print(f"{C.RED}Error: Directory not found: {directory_path}{C.ENDC}"); return
    all_files = sorted(glob.glob(os.path.join(directory_path, "*.bin")))
    files_to_process = [f for f in all_files if not (f.endswith(".0.bin") or f.endswith(".1.bin"))]
    processed_count = 0; print(f"Found {len(all_files)} total .bin files, processing {len(files_to_process)} relevant files.")
    for filepath in files_to_process: process_file_vNext(filepath, dump_hex, verbose=verbose); processed_count += 1
    if processed_count == 0: print(f"\n{C.YELLOW}No suitable .bin files found or processed in {directory_path}{C.ENDC}")
    else: print(f"\n{C.GREEN}Finished processing {processed_count} file(s).{C.ENDC}")

def get_notepad_tabstate_path():
    # (Function remains unchanged)
    localappdata = os.environ.get('LOCALAPPDATA')
    if localappdata:
        notepad_path = os.path.join(localappdata, r"Packages\Microsoft.WindowsNotepad_8wekyb3d8bbwe\LocalState\TabState")
        if os.path.isdir(notepad_path): return notepad_path
    return None

if __name__ == '__main__':
    # --- Timestamp and Location Context ---
    # (Section remains unchanged)
    print(f"# Script execution started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"# Location context: Upminster, England, United Kingdom")
    try:
        if pytz: user_tz_name = 'Europe/London'; user_tz = pytz.timezone(user_tz_name); context_dt = datetime.datetime.now(user_tz); print(f"{C.HEADER}--- Notepad Decoder --- Running at {context_dt.strftime('%Y-%m-%d %H:%M:%S %Z%z')} ({user_tz_name}) ---{C.ENDC}")
        else: raise Exception("pytz not available")
    except Exception as e:
        print(f"{C.YELLOW}Warning: Could not use specified timezone ({e}). Using system default timezone.{C.ENDC}")
        try: context_dt = datetime.datetime.now().astimezone(); tz_name = context_dt.tzname(); offset_str = context_dt.strftime('%z'); tz_display = f" {tz_name}" if tz_name else ""; print(f"{C.HEADER}--- Notepad Decoder --- Running at {context_dt.strftime('%Y-%m-%d %H:%M:%S')}{tz_display}{offset_str} ---{C.ENDC}")
        except Exception: print(f"{C.HEADER}--- Notepad Decoder --- Running at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (timezone unknown) ---{C.ENDC}")
    print("")

    # --- Argument Parsing ---
    # (Section remains unchanged)
    parser = argparse.ArgumentParser(description="Decode Notepad .bin files using primary and fallback logic.")
    parser.add_argument("-f", "--file", help="Path to a single .bin file.")
    parser.add_argument("-d", "--directory", help="Path to the directory containing .bin files (e.g., TabState). Default: Auto-detect.")
    parser.add_argument("--no-hex", action="store_true", help="Disable partial hex dump output for each file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug output.")
    args = parser.parse_args()

    dump_hex_enabled = not args.no_hex
    verbose_enabled = args.verbose

    # --- Execution Logic ---
    # (Section remains unchanged)
    if args.file:
        if os.path.isfile(args.file): print(f"Processing single file: {args.file}"); process_file_vNext(args.file, dump_hex_enabled, verbose=verbose_enabled)
        else: print(f"{C.RED}Error: Specified file not found: {args.file}{C.ENDC}"); sys.exit(1)
    else:
        target_directory = args.directory
        if not target_directory: print("No directory specified via -d, attempting to locate default Notepad TabState..."); target_directory = get_notepad_tabstate_path()
        if target_directory: print(f"Using directory: {target_directory}")
        else: print(f"\n{C.RED}Error: Could not automatically find TabState directory and none was specified via -d.{C.ENDC}"); parser.print_help(); sys.exit(1)
        if not os.path.isdir(target_directory): print(f"{C.RED}Error: Target directory not found: {target_directory}{C.ENDC}"); parser.print_help(); sys.exit(1)
        process_directory_vNext(target_directory, dump_hex_enabled, verbose=verbose_enabled)
