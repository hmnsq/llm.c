# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
আনযন sys
আনযন argparse

def read_numbers_from_file(file_path, col_start, col_end):
    যত্ন:
        numbers = []
        with open(file_path, 'r') ইৱ file:
            সাৰিগুলি = file.readlines()
            start_index = None
            বাবে i, সাৰি অন্তঃ enumerate(সাৰিগুলি):
                যদি "step    1/10" অন্তঃ সাৰি:
                    start_index = i
                    ভঙ্গ

            যদি start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                ঘূৰাই None

            # Read 10 rows starting from the identified start row
            বাবে সাৰি অন্তঃ সাৰিগুলি[start_index:start_index + 10]:
                # Extracting the specified columns
                সংখ্যা = ভিন্ন(সাৰি[col_start:col_end].strip())
                numbers.append(সংখ্যা)
        ঘূৰাই numbers
    অপৱাদ Exception ইৱ e:
        print(f"Error reading the file: {e}")
        ঘূৰাই None

def compare_numbers(read_মানসমূহ, fixed_মানসমূহ, percent_accuracy):
    বাবে i অন্তঃ range(len(read_মানসমূহ)):
        read_মান = read_মানসমূহ[i]
        fixed_মান = fixed_মানসমূহ[i]
        percent_difference = ((read_মান - fixed_মান) / fixed_মান) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        যদি abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            ঘূৰাই 1
    print("Success: All values are within the allowed accuracy.")
    ঘূৰাই 0

def main():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=সঁচা, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=পূর্ণাঙ্ক, required=সঁচা, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=পূর্ণাঙ্ক, required=সঁচা, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=ভিন্ন, required=সঁচা, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_মানসমূহ = read_numbers_from_file(args.file, args.col_start, args.col_end)
    যদি read_মানসমূহ is None:
        ঘূৰাই 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_মানসমূহ = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_মানসমূহ, fixed_মানসমূহ, args.percent_accuracy)
    ঘূৰাই result

যদি __name__ == "__main__":
    sys.exit(main())
