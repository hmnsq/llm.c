# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
استورد sys
استورد argparse

def read_numbers_from_file(file_path, col_start, col_end):
    حاول:
        numbers = []
        with open(file_path, 'r') كـ file:
            lines = file.readlines()
            start_index = None
            עבור i, line في enumerate(lines):
                אם "step    1/10" في line:
                    start_index = i
                    اكسر

            אם start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                החזר None

            # Read 10 rows starting from the identified start row
            עבור line في lines[start_index:start_index + 10]:
                # Extracting the specified columns
                number = عائم(line[col_start:col_end].strip())
                numbers.append(number)
        החזר numbers
    استثناء Exception كـ e:
        print(f"Error reading the file: {e}")
        החזר None

def compare_numbers(read_values, fixed_values, percent_accuracy):
    עבור i في range(len(read_values)):
        read_value = read_values[i]
        fixed_value = fixed_values[i]
        percent_difference = ((read_value - fixed_value) / fixed_value) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        אם abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            החזר 1
    print("Success: All values are within the allowed accuracy.")
    החזר 0

def main():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=אמת, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=صحيح, required=אמת, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=صحيح, required=אמת, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=عائم, required=אמת, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_values = read_numbers_from_file(args.file, args.col_start, args.col_end)
    אם read_values is None:
        החזר 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_values = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_values, fixed_values, args.percent_accuracy)
    החזר result

אם __name__ == "__main__":
    sys.exit(main())
