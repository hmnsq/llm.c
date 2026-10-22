# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
استورد sys
استورد argparse

def read_numbers_from_file(file_path, col_start, col_end):
    حاول:
        numbers = []
        with open(file_path, 'r') كـ file:
            أسطر = file.readlines()
            start_index = None
            لكل i, سطر في enumerate(أسطر):
                إذا "step    1/10" في سطر:
                    start_index = i
                    اكسر

            إذا start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                أرجع None

            # Read 10 rows starting from the identified start row
            لكل سطر في أسطر[start_index:start_index + 10]:
                # Extracting the specified columns
                رقم = عائم(سطر[col_start:col_end].strip())
                numbers.append(رقم)
        أرجع numbers
    استثناء Exception كـ e:
        print(f"Error reading the file: {e}")
        أرجع None

def compare_numbers(read_قيم, fixed_قيم, percent_accuracy):
    لكل i في range(len(read_قيم)):
        read_قيمة = read_قيم[i]
        fixed_قيمة = fixed_قيم[i]
        percent_difference = ((read_قيمة - fixed_قيمة) / fixed_قيمة) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        إذا abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            أرجع 1
    print("Success: All values are within the allowed accuracy.")
    أرجع 0

def main():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=صحيح, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=int, required=صحيح, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=int, required=صحيح, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=عائم, required=صحيح, help='Allowed percent accuracy for comparison')

    وسائط = parser.parse_وسائط()

    # Read numbers from file
    read_قيم = read_numbers_from_file(وسائط.file, وسائط.col_start, وسائط.col_end)
    إذا read_قيم is None:
        أرجع 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_قيم = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_قيم, fixed_قيم, وسائط.percent_accuracy)
    أرجع result

إذا __name__ == "__main__":
    sys.exit(main())
