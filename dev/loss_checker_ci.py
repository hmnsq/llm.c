# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
આયાત sys
આયાત argparse

def read_numbers_from_file(file_path, col_start, col_end):
    પ્રયાસ:
        numbers = []
        with open(file_path, 'r') ઇવ file:
            લીટીઓ = file.readlines()
            start_index = None
            માટે i, લીટી માં enumerate(લીટીઓ):
                જો "step    1/10" માં લીટી:
                    start_index = i
                    તોડો

            જો start_index is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                પાછા None

            # Read 10 rows starting from the identified start row
            માટે લીટી માં લીટીઓ[start_index:start_index + 10]:
                # Extracting the specified columns
                સંખ્યા = દશાંશ(લીટી[col_start:col_end].strip())
                numbers.append(સંખ્યા)
        પાછા numbers
    અપવાદ Exception ઇવ e:
        print(f"Error reading the file: {e}")
        પાછા None

def compare_numbers(read_કિંમતો, fixed_કિંમતો, percent_accuracy):
    માટે i માં range(len(read_કિંમતો)):
        read_કિંમત = read_કિંમતો[i]
        fixed_કિંમત = fixed_કિંમતો[i]
        percent_difference = ((read_કિંમત - fixed_કિંમત) / fixed_કિંમત) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        જો abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            પાછા 1
    print("Success: All values are within the allowed accuracy.")
    પાછા 0

def મુખ્ય():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=સાચું, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=પૂર્ણાંક, required=સાચું, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=પૂર્ણાંક, required=સાચું, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=દશાંશ, required=સાચું, help='Allowed percent accuracy for comparison')

    args = parser.parse_args()

    # Read numbers from file
    read_કિંમતો = read_numbers_from_file(args.file, args.col_start, args.col_end)
    જો read_કિંમતો is None:
        પાછા 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_કિંમતો = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    result = compare_numbers(read_કિંમતો, fixed_કિંમતો, args.percent_accuracy)
    પાછા result

જો __name__ == "__main__":
    sys.exit(મુખ્ય())
