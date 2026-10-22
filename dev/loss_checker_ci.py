# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
आयात sys
आयात argparse

def read_numbers_from_file(file_path, col_start, col_end):
    प्रयत्न:
        numbers = []
        with open(file_path, 'r') इव file:
            ओळी = file.readlines()
            start_index = None
            साठी i, ओळ मध्ये enumerate(ओळी):
                जर "step    1/10" मध्ये ओळ:
                    start_index = i
                    थांबा

            जर start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                परत None

            # Read 10 rows starting from the identified start row
            साठी ओळ मध्ये ओळी[start_index:start_index + 10]:
                # Extracting the specified columns
                संख्या = दशांश(ओळ[col_start:col_end].strip())
                numbers.append(संख्या)
        परत numbers
    अपवाद Exception इव e:
        print(f"Error reading the file: {e}")
        परत None

def compare_numbers(read_मूल्ये, fixed_मूल्ये, percent_accuracy):
    साठी i मध्ये range(len(read_मूल्ये)):
        read_मूल्य = read_मूल्ये[i]
        fixed_मूल्य = fixed_मूल्ये[i]
        percent_difference = ((read_मूल्य - fixed_मूल्य) / fixed_मूल्य) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        जर abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            परत 1
    print("Success: All values are within the allowed accuracy.")
    परत 0

def main():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=सत्य, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=पूर्णांक, required=सत्य, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=पूर्णांक, required=सत्य, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=दशांश, required=सत्य, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_मूल्ये = read_numbers_from_file(args.file, args.col_start, args.col_end)
    जर read_मूल्ये is None:
        परत 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_मूल्ये = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_मूल्ये, fixed_मूल्ये, args.percent_accuracy)
    परत result

जर __name__ == "__main__":
    sys.exit(main())
