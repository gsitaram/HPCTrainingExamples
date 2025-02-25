// Copyright (c) 2025 AMD HPC Application Performance Team
// Author: Bob Robey, Bob.Robey@amd.com
// MIT License

#include <iostream>

#include "HotScience.hh"

using namespace std;

int main(int argc, char *argv[]){

   HotScience myscienceclass;

   int N=10000;
   double *x = new double[N];

   for (int k = 0; k < N; k++){
      myscienceclass.compute(&x[k], N);
   }

   delete[] x;

   cout << "Array value is " << x[0] << endl;
   cout << "Finished calculation" << endl;
}
