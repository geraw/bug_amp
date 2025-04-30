# t-test analizingh

csv_ab_title_line = []

for case in storage_ab[0]:
    for i in storage_ab[0][case]:
        # for alg in storage_ab[case][i]:
            for val in csv_A_B_name:  # Include 'best', '5th', '10th' for TT & BF
                csv_ab_title_line.append(f'{case}_{i*N_TRAIN}k_{val}')
    csv_ab_title_line.append(f'{case}_{i*N_TRAIN}_diff')
    csv_ab_title_line.append(f'{case}_{i*N_TRAIN}_rel')

with open(csv_ab_filename, 'w', newline='') as csvfile:
    csv_ab_writer = csv.writer(csvfile)
    csv_ab_writer.writerow(csv_ab_title_line)
    csv_ab_writer.writerows(csv_ab_rows)


    from scipy.stats import ttest_rel

    all_cl_values = []
    all_bf_values = []
    csv_ab_rows = []
    csv_ab_title_line = []

    csv_ab_title_line.append('Case')
    for i in range(1, NUM_TO_CHECK + 1):
        csv_ab_title_line.append(f'{i*N_TRAIN}_statistic')
        csv_ab_title_line.append(f'{i*N_TRAIN}_p_value')
        csv_ab_title_line.append(f'{i*N_TRAIN}_significantly')

    csv_ab_writer.writerow(csv_ab_title_line)

    for case in csv_cases_name:
        cl_values = []
        bf_values = []
        csv_ab_row = []
        csv_ab_row.append(case)
        # Perform paired t-tests for each index
        for i in range(1, NUM_TO_CHECK + 1):

            for l in range(NUM_OF_TESTS):
                # Ensure the case and index exist in storage_ab
                cl = storage_ab[l][case][i]['CL']
                bf = storage_ab[l][case][i]['BF']
                cl_values.append(cl)
                bf_values.append(bf)
                all_cl_values.append(cl)
                all_bf_values.append(bf)

            # Check if we have enough data to perform the t-test
            if len(cl_values) >= 2:
                # Perform a paired t-test
                statistic, p_value = ttest_rel(cl_values, bf_values)
                csv_ab_row.append(statistic)
                csv_ab_row.append(p_value)
                # Analyze results
                print(f"\nIndex {i}k:")
                print(f"t-statistic: {statistic:.4f}, p-value: {p_value:.4f}")
                if p_value < 0.05:
                    if statistic > 0:
                        csv_ab_row.append('CL significantly than BF')
                        print("V CL performs significantly better than BF.")
                    else:
                        csv_ab_row.append('BF significantly than CL')
                        print("X BF performs significantly better than CL.")
                else:
                    csv_ab_row.append('No statistically')
                    print("No statistically significant difference between CL and BF.")
            else:
                csv_ab_row.append('Not enough data')
                print(f"\nIndex {i}: Not enough data to perform t-test.")

        csv_ab_rows.append(csv_ab_row)
        csv_ab_writer.writerow(csv_ab_row)

    csv_ab_row = []
    csv_ab_row.append('Total')
    # Perform the paired t-test if sufficient data is available
    if len(all_cl_values) >= 2:
        statistic, p_value = ttest_rel(all_cl_values, all_bf_values)
        csv_ab_row.append(statistic)
        csv_ab_row.append(p_value)
        print("\nOverall Comparison:")
        print(f"t-statistic: {statistic:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            if statistic > 0:
                csv_ab_row.append('CL significantly than BF')
                print("V CL performs significantly better than BF.")
            else:
                csv_ab_row.append('BF significantly than CL')
                print("X BF performs significantly better than CL.")
        else:
            csv_ab_row.append('No statistically')
            print("No statistically significant difference between CL and BF.")
    else:
        csv_ab_row.append('Not enough data')
        print(f"\nIndex {i}: Not enough data to perform t-test.")
    csv_ab_rows.append(csv_ab_row)
    csv_ab_writer.writerow(csv_ab_row)




