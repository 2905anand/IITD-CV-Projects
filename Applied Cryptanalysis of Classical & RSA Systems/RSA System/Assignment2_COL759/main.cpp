#include <bits/stdc++.h>
#include <stdio.h>
#include <gmp.h>
#include <iostream>
#include <chrono>
#include "QuadraticSieve.h"


int main() {

    freopen("output.txt", "w", stdout);
    auto start = std::chrono::high_resolution_clock::now();
    /*/
     //! GIVEN TESTCASES ----


    1.input (19):  5455825547824891739
        output: a=4670642897 b=1168110187
        [ 1:04 seconds - [-500000, 500000] ]
        [ 0:31 seconds - [-100000, 100000] ]
        [ 0:01 Minutes - (-88620, 88620) ]


    2.input (24):  165050667374457919300349
        output: a=1058936939   b=155864491355191
        [ 0:21 Minutes - [-581000, 581000] ]


    3.input (26):  73122429608667816341022991
        output: a=147003417493    b=497419929792787
        [ 1:45 Minutes - [-1536426, 1536426] ]


    4.input (22):  9703841818924589184253
        output: a=163 b=59532771895242878431
        [ 1:33 Minutes - [-500000, 500000] ]
        [ 0:16 Minutes - [-363000, 363000] ]
        [ 0:10 Minutes - [-318402, 318402] ]


    5.input (28):  5486150758706356557450962117
        output: a=584081353   b=9392785320962568989)
        [ 7:05 Minutes - (-4000000, 4000000) ]
        [ 5:45 Minutes - (-2500000, 2500000) ]
    */

    mpz_class n("5486150758706356557450962117");

    int threads = 8;
    QuadraticSieve q(n, threads);

    std::cout << "\nSize of the Factor Base is = " << q.factorBase.size() << "\n";
    std::cout << "B-smooth Value = " << q.factorBase.back() << "\n\n";


    // Base case when it is square of a prime number.
    if(mpz_perfect_square_p(n.get_mpz_t())){
        mpz_class sqRoot = 1;
        mpz_sqrt(sqRoot.get_mpz_t(), n.get_mpz_t());
        std::cout << n << " is a perfect square.\n";
        std::cout << "Factors are " << sqRoot <<" And " << sqRoot << "\n";
        return 0;
    }

    int SieveSize = (int) pow(q.factorBase.size(), 2);
    int left = (int)(-1) * (SieveSize / 2);
    int right = (int)(SieveSize / 2);
    std::cout << "Sieving from " << left << " to " << right << "\n\n";

    std::vector<std::vector<mpz_class>> mat;
    q.sieve(left, right, mat);
    std::cout << "\n\nMatrix Size = " << mat.size() << "\n\n";

    
    std::vector<std::vector<mpz_class>> matrix(mat.size());

    for (int i = 0; i < mat.size(); i++)
    {
        for (int j = 2; j < mat[i].size(); j++)
        {
            // std::cout << mat[i][j] << " ";
            matrix[i].push_back(mat[i][j]);
        }
    }

        std::vector<std::vector<int>> indexes = q.findLinearDependencies(matrix);

        std::cout << "Found " << indexes.size() << " dependencies\n\n";

        // for (int i = 0; i < indexes.size(); i++){
        //     for (int j = 0; j < indexes[i].size(); j++){
        //         std::cout << indexes[i][j] << " ";
        //     }
        //     std::cout  << "\n";
        // }

        for (int i = 0; i < indexes.size(); i++)
        {

            std::vector<mpz_class> qValues;
            for (int j = 0; j < indexes[i].size(); j++)
            {
                qValues.push_back(mat[indexes[i][j]][0]);
                qValues.push_back(mat[indexes[i][j]][1]);
            }

            std::vector<mpz_class> factors = q.getDivisors(qValues);

            if (((factors[0] != 1 && factors[1] != 1) && (factors[0] != 1 && factors[1] != n) && (factors[0] != n && factors[1] != 1)))
            {   
                std::cout << "--------------------------------\n";
                std::cout << "THE FINAL FACTORS ARE " << factors[0] << " AND/OR " << factors[1] << "\n";
                std::cout << "--------------------------------\n";
                break;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "\n\n--------------------------------\n";
        std::cout << "Time taken: " << duration.count() << " seconds.\n";
        std::cout << "--------------------------------\n";

        return 0;
}

