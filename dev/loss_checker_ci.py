# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
आनयन sys
आनयन argparse

def read_numbers_from_file(file_path, col_start, col_end):
    यत्न:
        numbers = []
        with open(file_path, 'r') इव file:
            पङ्क्तयः = file.readlines()
            start_index = None
            प्रति i, पङ्क्तिः अन्तः enumerate(पङ्क्तयः):
                जइ "step    1/10" अन्तः पङ्क्तिः:
                    start_index = i
                    भङ्ग

            जइ start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                प्रत्यागम None

            # Read 10 rows starting from the identified start row
            प्रति पङ्क्तिः अन्तः पङ्क्तयः[start_index:start_index + 10]:
                # Extracting the specified columns
                सङ्ख्या = भिन्न(पङ्क्तिः[col_start:col_end].strip())
                numbers.append(सङ्ख्या)
        प्रत्यागम numbers
    अपवाद Exception इव e:
        print(f"Error reading the file: {e}")
        प्रत्यागम None

def compare_numbers(read_मानानि, fixed_मानानि, percent_accuracy):
    प्रति i अन्तः range(len(read_मानानि)):
        read_मानम् = read_मानानि[i]
        fixed_मानम् = fixed_मानानि[i]
        percent_difference = ((read_मानम् - fixed_मानम्) / fixed_मानम्) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        जइ abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            प्रत्यागम 1
    print("Success: All values are within the allowed accuracy.")
    प्रत्यागम 0

def main():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=सच्च, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=पूर्णाङ्क, required=सच्च, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=पूर्णाङ्क, required=सच्च, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=भिन्न, required=सच्च, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_मानानि = read_numbers_from_file(args.file, args.col_start, args.col_end)
    जइ read_मानानि is None:
        प्रत्यागम 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_मानानि = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_मानानि, fixed_मानानि, args.percent_accuracy)
    प्रत्यागम result

जइ __name__ == "__main__":
    sys.exit(main())
