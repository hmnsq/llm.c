# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
ഇറക്കുമതി sys
ഇറക്കുമതി argparse

def read_numbers_from_file(file_path, col_start, col_end):
    ശ്രമിക്കുക:
        numbers = []
        with open(file_path, 'r') ഇവ file:
            വരികൾ = file.readlines()
            start_index = None
            വേണ്ടി i, വരി ഇൽ enumerate(വരികൾ):
                എങ്കിൽ "step    1/10" ഇൽ വരി:
                    start_index = i
                    നിർത്തുക

            എങ്കിൽ start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                തിരികെ None

            # Read 10 rows starting from the identified start row
            വേണ്ടി വരി ഇൽ വരികൾ[start_index:start_index + 10]:
                # Extracting the specified columns
                സംഖ്യ = ദശാംശം(വരി[col_start:col_end].strip())
                numbers.append(സംഖ്യ)
        തിരികെ numbers
    അപവാദം Exception ഇവ e:
        print(f"Error reading the file: {e}")
        തിരികെ None

def compare_numbers(read_മൂല്യങ്ങൾ, fixed_മൂല്യങ്ങൾ, percent_accuracy):
    വേണ്ടി i ഇൽ range(len(read_മൂല്യങ്ങൾ)):
        read_മൂല്യം = read_മൂല്യങ്ങൾ[i]
        fixed_മൂല്യം = fixed_മൂല്യങ്ങൾ[i]
        percent_difference = ((read_മൂല്യം - fixed_മൂല്യം) / fixed_മൂല്യം) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        എങ്കിൽ abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            തിരികെ 1
    print("Success: All values are within the allowed accuracy.")
    തിരികെ 0

def പ്രധാന():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=സത്യം, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=പൂർണ്ണസംഖ്യ, required=സത്യം, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=പൂർണ്ണസംഖ്യ, required=സത്യം, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=ദശാംശം, required=സത്യം, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_മൂല്യങ്ങൾ = read_numbers_from_file(args.file, args.col_start, args.col_end)
    എങ്കിൽ read_മൂല്യങ്ങൾ is None:
        തിരികെ 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_മൂല്യങ്ങൾ = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_മൂല്യങ്ങൾ, fixed_മൂല്യങ്ങൾ, args.percent_accuracy)
    തിരികെ result

എങ്കിൽ __name__ == "__main__":
    sys.exit(പ്രധാന())
