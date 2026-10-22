# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
واردکن sys
واردکن argparse

def read_numbers_from_file(file_path, col_start, col_end):
    تلاش:
        numbers = []
        with open(file_path, 'r') بهعنوان file:
            سطرها = file.readlines()
            start_index = None
            برای i, سطر در enumerate(سطرها):
                اگر "step    1/10" در سطر:
                    start_index = i
                    بشکن

            اگر start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                بازگشت None

            # Read 10 rows starting from the identified start row
            برای سطر در سطرها[start_index:start_index + 10]:
                # Extracting the specified columns
                عدد = اعشاری(سطر[col_start:col_end].strip())
                numbers.append(عدد)
        بازگشت numbers
    استثنا Exception بهعنوان e:
        print(f"Error reading the file: {e}")
        بازگشت None

def compare_numbers(read_مقادیر, fixed_مقادیر, percent_accuracy):
    برای i در range(len(read_مقادیر)):
        read_مقدار = read_مقادیر[i]
        fixed_مقدار = fixed_مقادیر[i]
        percent_difference = ((read_مقدار - fixed_مقدار) / fixed_مقدار) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        اگر abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            بازگشت 1
    print("Success: All values are within the allowed accuracy.")
    بازگشت 0

def اصلی():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=درست, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=صحیح, required=درست, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=صحیح, required=درست, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=اعشاری, required=درست, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_مقادیر = read_numbers_from_file(args.file, args.col_start, args.col_end)
    اگر read_مقادیر is None:
        بازگشت 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_مقادیر = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_مقادیر, fixed_مقادیر, args.percent_accuracy)
    بازگشت result

اگر __name__ == "__main__":
    sys.exit(اصلی())
