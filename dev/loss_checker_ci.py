# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
आनयन sys
आनयन argparse

def read_numbers_from_file(file_path, col_start, col_end):
    यत्न:
        numbers = []
        with open(file_path, 'r') इव file:
            पङ्क्तिहरू = file.readlines()
            start_index = None
            लागि i, पङ्क्ति अन्तः enumerate(पङ्क्तिहरू):
                यदि "step    1/10" अन्तः पङ्क्ति:
                    start_index = i
                    रोक

            यदि start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                फर्काऊ None

            # Read 10 rows starting from the identified start row
            लागि पङ्क्ति अन्तः पङ्क्तिहरू[start_index:start_index + 10]:
                # Extracting the specified columns
                सङ्ख्या = भिन्न(पङ्क्ति[col_start:col_end].strip())
                numbers.append(सङ्ख्या)
        फर्काऊ numbers
    अपवाद Exception इव e:
        print(f"Error reading the file: {e}")
        फर्काऊ None

def compare_numbers(read_मानहरू, fixed_मानहरू, percent_accuracy):
    लागि i अन्तः range(len(read_मानहरू)):
        read_मान = read_मानहरू[i]
        fixed_मान = fixed_मानहरू[i]
        percent_difference = ((read_मान - fixed_मान) / fixed_मान) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        यदि abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            फर्काऊ 1
    print("Success: All values are within the allowed accuracy.")
    फर्काऊ 0

def main():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=सत्य, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=पूर्णाङ्क, required=सत्य, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=पूर्णाङ्क, required=सत्य, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=भिन्न, required=सत्य, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_मानहरू = read_numbers_from_file(args.file, args.col_start, args.col_end)
    यदि read_मानहरू is None:
        फर्काऊ 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_मानहरू = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_मानहरू, fixed_मानहरू, args.percent_accuracy)
    फर्काऊ result

यदि __name__ == "__main__":
    sys.exit(main())
