# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
ଆମଦାନୀ sys
ଆମଦାନୀ argparse

def read_numbers_from_file(file_path, col_start, col_end):
    ଚେଷ୍ଟା:
        numbers = []
        with open(file_path, 'r') ଇଵ file:
            ଧାଡ଼ିଗୁଡ଼ିକ = file.readlines()
            start_index = None
            ପାଇଁ i, ଧାଡ଼ି ଭିତରେ enumerate(ଧାଡ଼ିଗୁଡ଼ିକ):
                ଯଦି "step    1/10" ଭିତରେ ଧାଡ଼ି:
                    start_index = i
                    ଭାଙ୍ଗ

            ଯଦି start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                ଫେରନ୍ତୁ None

            # Read 10 rows starting from the identified start row
            ପାଇଁ ଧାଡ଼ି ଭିତରେ ଧାଡ଼ିଗୁଡ଼ିକ[start_index:start_index + 10]:
                # Extracting the specified columns
                ସଂଖ୍ୟା = ଦଶମିକ(ଧାଡ଼ି[col_start:col_end].strip())
                numbers.append(ସଂଖ୍ୟା)
        ଫେରନ୍ତୁ numbers
    ବ୍ୟତିକ୍ରମ Exception ଇଵ e:
        print(f"Error reading the file: {e}")
        ଫେରନ୍ତୁ None

def compare_numbers(read_ମୂଲ୍ୟଗୁଡ଼ିକ, fixed_ମୂଲ୍ୟଗୁଡ଼ିକ, percent_accuracy):
    ପାଇଁ i ଭିତରେ range(len(read_ମୂଲ୍ୟଗୁଡ଼ିକ)):
        read_ମୂଲ୍ୟ = read_ମୂଲ୍ୟଗୁଡ଼ିକ[i]
        fixed_ମୂଲ୍ୟ = fixed_ମୂଲ୍ୟଗୁଡ଼ିକ[i]
        percent_difference = ((read_ମୂଲ୍ୟ - fixed_ମୂଲ୍ୟ) / fixed_ମୂଲ୍ୟ) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        ଯଦି abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            ଫେରନ୍ତୁ 1
    print("Success: All values are within the allowed accuracy.")
    ଫେରନ୍ତୁ 0

def ମୁଖ୍ୟ():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=ସତ୍ୟ, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, required=ସତ୍ୟ, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, required=ସତ୍ୟ, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=ଦଶମିକ, required=ସତ୍ୟ, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_ମୂଲ୍ୟଗୁଡ଼ିକ = read_numbers_from_file(args.file, args.col_start, args.col_end)
    ଯଦି read_ମୂଲ୍ୟଗୁଡ଼ିକ is None:
        ଫେରନ୍ତୁ 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_ମୂଲ୍ୟଗୁଡ଼ିକ = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_ମୂଲ୍ୟଗୁଡ଼ିକ, fixed_ମୂଲ୍ୟଗୁଡ଼ିକ, args.percent_accuracy)
    ଫେରନ୍ତୁ result

ଯଦି __name__ == "__main__":
    sys.exit(ମୁଖ୍ୟ())
