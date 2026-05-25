// Test if armadillo's swap_rows/swap_cols on complex matrices works correctly.
#include <armadillo>
#include <iostream>
using namespace arma;
using cmpx = std::complex<double>;
int main() {
    cx_mat A(4,4, fill::zeros);
    for (int i=0; i<4; i++) for (int j=0; j<4; j++) A(i,j) = cmpx(i*10+j, 0);
    std::cout << "Before:\n" << A << "\n";
    A.swap_cols(1, 3);
    std::cout << "After swap_cols(1,3):\n" << A << "\n";
    A.swap_rows(1, 3);
    std::cout << "After swap_rows(1,3):\n" << A << "\n";

    // Diagonal swap test (relevant case)
    cx_mat D = diagmat(cx_vec({cmpx(1,0), cmpx(0,0), cmpx(1,0), cmpx(0,0)}));
    std::cout << "Diag before:\n" << D << "\n";
    D.swap_cols(1, 3);
    D.swap_rows(1, 3);
    std::cout << "Diag after swap (1,3):\n" << D << "\n";
    return 0;
}
