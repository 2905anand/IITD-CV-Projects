#include <stdio.h>
#include <bits/stdc++.h>
#include <gmp.h>
#include <gmpxx.h>
#include <iostream>
#include <condition_variable>
#include <thread>



class QuadraticSieve {
public:
    mpz_class number;
    std::vector<int> factorBase;
    std::mutex mtx;  // Mutex for matrix
    std::mutex printMutex; // Mutex for printing
    std::condition_variable cv;
    bool stop_processing = false;
    int threads;

    QuadraticSieve(mpz_class a, int t) : number(a), threads(t)  {
        int factorBaseSize = static_cast<int>(exp(0.474 * sqrt(log(a.get_d()) * (log(log(a.get_d())) ))));
        factorBase = generateFactorBase(factorBaseSize);
    };

    // Checking if valid legendraSymbol exits for n, and prime p
    bool isValidLegenedraSymbol(mpz_class n, mpz_class p);

    std::vector<int> generateFactorBase(int size);

    // Multithreaded sieve method
    void sieve(int left, int right, std::vector<std::vector<mpz_class>> &matrix);

    // Worker function for each thread
    void sieve_worker(int start, int end, std::vector<std::vector<mpz_class>> &matrix);

    std::vector<mpz_class> getDivisors(std::vector<mpz_class> &zeroRows);

    std::vector<std::vector<int>> findLinearDependencies(std::vector<std::vector<mpz_class>> &matrix);
};

std::vector<mpz_class> QuadraticSieve::getDivisors(std::vector<mpz_class> &zeroRows){
    mpz_class Q_x_mul = 1;
    mpz_class a_mul = 1;
    for (int i = 0; i < zeroRows.size(); i+=2){
        Q_x_mul = (Q_x_mul * zeroRows[i]);
        a_mul = (a_mul * zeroRows[i + 1]);
    }

    mpz_class x = a_mul % number;
    mpz_class y = 1;

    if(mpz_sgn(Q_x_mul.get_mpz_t()) < 0){
        return {1, 1};
    }

    mpz_sqrt(y.get_mpz_t(), Q_x_mul.get_mpz_t());
    y = y % number;

    mpz_class result1 = 0, result2 = 0;
    mpz_class xMinusY(x - y);
    mpz_class xPlusY(x + y);
    mpz_gcd(result1.get_mpz_t(), xMinusY.get_mpz_t(), number.get_mpz_t());
    mpz_gcd(result2.get_mpz_t(), xPlusY.get_mpz_t(), number.get_mpz_t());

    return { result1, result2};
}

// Worker function to sieve a sub-range in parallel
void QuadraticSieve::sieve_worker(int start, int end, std::vector<std::vector<mpz_class>> &matrix) {

    std::unique_lock<std::mutex> printLock(printMutex);

    std::thread::id this_id = std::this_thread::get_id();
    std::cout << "Thread ID - " << this_id << "  "
              << "Sieving in Range [ " << start << ", " << end << " ] \n";
    printLock.unlock();

    mpz_class m;
    mpz_sqrt(m.get_mpz_t(), number.get_mpz_t());  // sqrt(n)

    std::vector<std::vector<mpz_class>> local_matrix;

    for (int i = start; i <= end && !stop_processing; i++) {
        mpz_class Q_x = ((i + m) * (i + m)) - number;
        std::vector<mpz_class> exponents(factorBase.size() + 2);  // for Q(x) and a=(i+m)

        exponents[0] = Q_x;  // Store Q(x)
        mpz_class a = (i + m);
        exponents[1] = a;

        if (mpz_sgn(Q_x.get_mpz_t()) < 0) {
            exponents[2] = -1;  // Exponent for -1
            mpz_abs(Q_x.get_mpz_t(), Q_x.get_mpz_t());
        } else {
            exponents[2] = 0;  // No contribution from -1
        }

        mpz_class remainder = Q_x;
        bool is_b_smooth = true;

        for (size_t j = 0; j < factorBase.size()&&!stop_processing; j++) {
            int prime = factorBase[j];
            int exponent = 0;

            while (!stop_processing && mpz_divisible_ui_p(remainder.get_mpz_t(), prime)) {
                mpz_divexact_ui(remainder.get_mpz_t(), remainder.get_mpz_t(), prime);
                exponent++;
                if (exponent > factorBase.size()) {
                    is_b_smooth = false;
                    break;
                }
            }

            if (!is_b_smooth) break;

            exponents[j + 2] = exponent % 2;  // Store the exponent mod 2
        }

        if (remainder != 1) {
            is_b_smooth = false;
        }

        if (is_b_smooth) {
            local_matrix.push_back(exponents);
        }

        // Locking to safely modify shared matrix and check size
        std::unique_lock<std::mutex> lock(mtx);
        int sieveSize = factorBase.size() + 200;
        // std::cout << "Inserting Started \n";
        if (matrix.size() >= sieveSize)
        {
            stop_processing = true;  // Signal other threads to stop
            return;  // Exit this thread early
        }

        if (local_matrix.size() + matrix.size() > sieveSize) {
            size_t remaining_slots = sieveSize - matrix.size();
            matrix.insert(matrix.end(), local_matrix.begin(), local_matrix.begin() + remaining_slots);
            stop_processing = true;
            return;
        }
    }

    // Lock the matrix and append local results
    std::unique_lock<std::mutex> lock(mtx);
    matrix.insert(matrix.end(), local_matrix.begin(), local_matrix.end());
    lock.unlock();
}

// The main sieve function that creates threads and splits the range
void QuadraticSieve::sieve(int left, int right, std::vector<std::vector<mpz_class>> &matrix) {
    int num_threads = threads;  // Number of threads (you can adjust this)
    int range_size = (right - left) / num_threads;  // Divide the range

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = left + i * range_size;
        int end = (i == num_threads - 1) ? right : start + range_size - 1;
        threads.push_back(std::thread(&QuadraticSieve::sieve_worker, this, start, end, std::ref(matrix)));
    }

    // Join all threads
    for (auto &t : threads) {
        t.join();
    }
}


bool QuadraticSieve::isValidLegenedraSymbol(mpz_class n, mpz_class p){
    int legendre = mpz_legendre(n.get_mpz_t(), p.get_mpz_t());
    if(legendre==1)
        return true;
    return false;
}

std::vector<int> QuadraticSieve::generateFactorBase(int size){
    // Here size represent the size of the factorbase array
    std::vector<int> factorBase;
    factorBase.push_back(-1);
    std::vector<bool> isPrime(1500000, true);

    for (int i = 2; i < isPrime.size() && factorBase.size() <= size; i++){
        if(isPrime[i] == true && isValidLegenedraSymbol(number, i)){
            factorBase.push_back(i);
        }
        if(isPrime[i]){
            for (int j = i*i; j < isPrime.size(); j += i ){
                isPrime[j] = false;
            }
        }
    }
    return factorBase;
}


std::vector<std::vector<int>> QuadraticSieve::findLinearDependencies(std::vector<std::vector<mpz_class>>& matrix) {
    int rows_plus_200 = matrix.size(); // Number of rows
    int t = matrix[0].size();      // Number of columns

    // Initialize dependency tracking: each row starts as its own basis
    std::vector<std::vector<int>> dependencies(rows_plus_200, std::vector<int>(rows_plus_200, 0));
    for (int i = 0; i < rows_plus_200; ++i) {
        dependencies[i][i] = 1;
    }

    // Perform Gaussian elimination starting from the third column (index 2)
    for (int col = 2; col < t; ++col) { // Start from column 2
        int pivot_row = -1;
        for (int row = col; row < rows_plus_200; ++row) {
            if (matrix[row][col] == 1) {
                pivot_row = row;
                break;
            }
        }

        if (pivot_row == -1) {
            continue; // No pivot in this column
        }

        if (pivot_row != col) {
            // Swap rows in matrix
            swap(matrix[col], matrix[pivot_row]);
            // Swap dependency rows
            swap(dependencies[col], dependencies[pivot_row]);
        }

        // Eliminate other rows
        for (int row = 0; row < rows_plus_200; ++row) {
            if (row != col && matrix[row][col] == 1) {
                // XOR the pivot row into this row
                for (int i = 2; i < t; ++i) { // Ignore the first two columns
                    matrix[row][i] ^= matrix[col][i];
                }
                for (int i = 0; i < rows_plus_200; ++i) {
                    dependencies[row][i] ^= dependencies[col][i];
                }
            }
        }
    }

    // Identify dependencies: rows that are all zeros (ignoring first two columns)
    std::vector<std::vector<int>> dependencies_found;
    for (int row = 0; row < rows_plus_200; ++row) {
        bool all_zeros = true;
        for (int col = 2; col < t; ++col) { // Check from the third column
            if (matrix[row][col] != 0) {
                all_zeros = false;
                break;
            }
        }
        if (all_zeros) {
            std::vector<int> dependent_rows;
            for (int i = 0; i < rows_plus_200; ++i) {
                if (dependencies[row][i] == 1) {
                    dependent_rows.push_back(i);
                }
            }
            if (!dependent_rows.empty()) {
                dependencies_found.push_back(dependent_rows);
            }
        }
    }

    return dependencies_found;
}

