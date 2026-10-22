# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
আমদানি sys
আমদানি argparse

def read_numbers_from_file(file_path, col_start, col_end):
    চেষ্টা:
        numbers = []
        with open(file_path, 'r') ইব file:
            সারিগুলি = file.readlines()
            start_সূচক = None
            জন্য i, সারি মধ্যে enumerate(সারিগুলি):
                যদি "step    1/10" মধ্যে সারি:
                    start_সূচক = i
                    বিরতি

            যদি start_সূচক is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                ফেরত None

            # Read 10 rows starting from the identified start row
            জন্য সারি মধ্যে সারিগুলি[start_সূচক:start_সূচক + 10]:
                # Extracting the specified columns
                সংখ্যা = দশমিক(সারি[col_start:col_end].strip())
                numbers.append(সংখ্যা)
        ফেরত numbers
    ব্যতিক্রম Exception ইব e:
        print(f"Error reading the file: {e}")
        ফেরত None

def compare_numbers(read_মানসমূহ, fixed_মানসমূহ, percent_accuracy):
    জন্য i মধ্যে range(len(read_মানসমূহ)):
        read_মান = read_মানসমূহ[i]
        fixed_মান = fixed_মানসমূহ[i]
        percent_difference = ((read_মান - fixed_মান) / fixed_মান) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        যদি abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            ফেরত 1
    print("Success: All values are within the allowed accuracy.")
    ফেরত 0

def প্রধান():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=সত্য, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=পূর্ণসংখ্যা, required=সত্য, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=পূর্ণসংখ্যা, required=সত্য, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=দশমিক, required=সত্য, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_মানসমূহ = read_numbers_from_file(args.file, args.col_start, args.col_end)
    যদি read_মানসমূহ is None:
        ফেরত 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_মানসমূহ = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    ফলাফল = compare_numbers(read_মানসমূহ, fixed_মানসমূহ, args.percent_accuracy)
    ফেরত ফলাফল

যদি __name__ == "__main__":
    sys.exit(প্রধান())
