# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
import sys
import argparse

def read_numbers_from_file(file_path, col_start, col_end):
    try:
        numbers = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            start_index = None
            ཆེད་དུ i, line anta_h enumerate(lines):
                གལ་ཏེ "step    1/10" anta_h line:
                    start_index = i
                    bha_nga

            གལ་ཏེ start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                ཕྱིར None

            # Read 10 rows starting from the identified start row
            ཆེད་དུ line anta_h lines[start_index:start_index + 10]:
                # Extracting the specified columns
                number = float(line[col_start:col_end].strip())
                numbers.append(number)
        ཕྱིར numbers
    except Exception as e:
        print(f"Error reading the file: {e}")
        ཕྱིར None

def compare_numbers(read_values, fixed_values, percent_accuracy):
    ཆེད་དུ i anta_h range(len(read_values)):
        read_value = read_values[i]
        fixed_value = fixed_values[i]
        percent_difference = ((read_value - fixed_value) / fixed_value) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        གལ་ཏེ abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            ཕྱིར 1
    print("Success: All values are within the allowed accuracy.")
    ཕྱིར 0

def གཙོ():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=True, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=ཧྲིལ, required=True, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=ཧྲིལ, required=True, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=float, required=True, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_values = read_numbers_from_file(args.file, args.col_start, args.col_end)
    གལ་ཏེ read_values is None:
        ཕྱིར 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_values = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_values, fixed_values, args.percent_accuracy)
    ཕྱིར result

གལ་ཏེ __name__ == "__main__":
    sys.exit(གཙོ())
