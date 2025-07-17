#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>

// Spiner EOS
#include <singularity-eos/eos/eos.hpp>
#include <singularity-eos/eos/eos_SpinerEOSDependsRhoT.hpp>

using namespace singularity;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " sp5_file.h5 matid\n";
    return 1;
  }

  std::string sp5_file = argv[1];
  int matid = std::atoi(argv[2]);
  double rho = 0.1;
  double sie = 1.0e12;

  singularity::SpinerEOSDependsRhoT eos(sp5_file, matid);
  singularity::TableStatus status;

  auto start = std::chrono::high_resolution_clock::now();
  double temperature = eos.TemperatureFromDensityInternalEnergy(rho, sie);
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(end - start).count();

  eos.DebugPrint();

  std::cout << "Temperature = " << temperature << " K\n";
  std::cout << "Elapsed time = " << elapsed << " s\n";

  return 0;
}
