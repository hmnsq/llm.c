# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
இறக்குமதி sys
இறக்குமதி argparse

def read_numbers_from_file(file_path, col_start, col_end):
    முயல்:
        numbers = []
        with open(file_path, 'r') இவ file:
            வரிகள் = file.readlines()
            start_index = None
            க்கு i, வரி உள் enumerate(வரிகள்):
                எனில் "step    1/10" உள் வரி:
                    start_index = i
                    நிறுத்து

            எனில் start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                திரும்பு None

            # Read 10 rows starting from the identified start row
            க்கு வரி உள் வரிகள்[start_index:start_index + 10]:
                # Extracting the specified columns
                எண் = தசம(வரி[col_start:col_end].strip())
                numbers.append(எண்)
        திரும்பு numbers
    விதிவிலக்கு Exception இவ e:
        print(f"Error reading the file: {e}")
        திரும்பு None

def compare_numbers(read_மதிப்புகள், fixed_மதிப்புகள், percent_accuracy):
    க்கு i உள் range(len(read_மதிப்புகள்)):
        read_மதிப்பு = read_மதிப்புகள்[i]
        fixed_மதிப்பு = fixed_மதிப்புகள்[i]
        percent_difference = ((read_மதிப்பு - fixed_மதிப்பு) / fixed_மதிப்பு) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        எனில் abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            திரும்பு 1
    print("Success: All values are within the allowed accuracy.")
    திரும்பு 0

def முதன்மை():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=உண்மை, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=முழு, required=உண்மை, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=முழு, required=உண்மை, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=தசம, required=உண்மை, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_மதிப்புகள் = read_numbers_from_file(args.file, args.col_start, args.col_end)
    எனில் read_மதிப்புகள் is None:
        திரும்பு 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_மதிப்புகள் = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_மதிப்புகள், fixed_மதிப்புகள், args.percent_accuracy)
    திரும்பு result

எனில் __name__ == "__main__":
    sys.exit(முதன்மை())
