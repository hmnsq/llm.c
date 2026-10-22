# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
ಆನಯನ sys
ಆನಯನ argparse

def read_numbers_from_file(file_path, col_start, col_end):
    ಯತ್ನ:
        numbers = []
        with open(file_path, 'r') ಇವ file:
            ಸಾಲುಗಳು = file.readlines()
            start_index = None
            ಪ್ರತಿ i, ಸಾಲು ಅನ್ತಃ enumerate(ಸಾಲುಗಳು):
                ಯದಿ "step    1/10" ಅನ್ತಃ ಸಾಲು:
                    start_index = i
                    ಭಙ್ಗ

            ಯದಿ start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                ಪ್ರತ್ಯಾಗಮ None

            # Read 10 rows starting from the identified start row
            ಪ್ರತಿ ಸಾಲು ಅನ್ತಃ ಸಾಲುಗಳು[start_index:start_index + 10]:
                # Extracting the specified columns
                ಸಂಖ್ಯೆ = ಭಿನ್ನ(ಸಾಲು[col_start:col_end].strip())
                numbers.append(ಸಂಖ್ಯೆ)
        ಪ್ರತ್ಯಾಗಮ numbers
    ಅಪವಾದ Exception ಇವ e:
        print(f"Error reading the file: {e}")
        ಪ್ರತ್ಯಾಗಮ None

def compare_numbers(read_ಮೌಲ್ಯಗಳು, fixed_ಮೌಲ್ಯಗಳು, percent_accuracy):
    ಪ್ರತಿ i ಅನ್ತಃ range(len(read_ಮೌಲ್ಯಗಳು)):
        read_ಮೌಲ್ಯ = read_ಮೌಲ್ಯಗಳು[i]
        fixed_ಮೌಲ್ಯ = fixed_ಮೌಲ್ಯಗಳು[i]
        percent_difference = ((read_ಮೌಲ್ಯ - fixed_ಮೌಲ್ಯ) / fixed_ಮೌಲ್ಯ) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        ಯದಿ abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            ಪ್ರತ್ಯಾಗಮ 1
    print("Success: All values are within the allowed accuracy.")
    ಪ್ರತ್ಯಾಗಮ 0

def main():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=ಸತ್ಯ, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=ಪೂರ್ಣಾಙ್ಕ, required=ಸತ್ಯ, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=ಪೂರ್ಣಾಙ್ಕ, required=ಸತ್ಯ, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=ಭಿನ್ನ, required=ಸತ್ಯ, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_ಮೌಲ್ಯಗಳು = read_numbers_from_file(args.file, args.col_start, args.col_end)
    ಯದಿ read_ಮೌಲ್ಯಗಳು is None:
        ಪ್ರತ್ಯಾಗಮ 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_ಮೌಲ್ಯಗಳು = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_ಮೌಲ್ಯಗಳು, fixed_ಮೌಲ್ಯಗಳು, args.percent_accuracy)
    ಪ್ರತ್ಯಾಗಮ result

ಯದಿ __name__ == "__main__":
    sys.exit(main())
