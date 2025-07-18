/*
  This script has the following capabilities:
  1. Compares the performance of SpinerEOSDependsRhoT and SpinerEOSDependsRhoSie without transformations
  2. Compares the performance of SpinerEOSDependsRhoT and SpinerEosDependsRhoSie with transformations 
  3. Compares them both to EOSPAC

  We test the following:
  1. P(rho, T) and e(rho, T) lookups on a log-regular grid of density-temperature points designed to cover the entire domain
  2. P(rho, T) and e(rho, T) lookups on the exact pairs of density-temperature points that correspond to the SESAME grid
  3. P(rho, e(rho, T)) and T(rho, e(rho, T)) lookups on the SESAME grid points

  Example usage: ./eos_benchmark ./output_dir 100 100 ./tables/my_eos.sp5 2020 3720

  Authors: Erin O'Neil and Joshua Basabe
 */

#ifdef _SPINER_USE_HDF_
#ifdef _SINGULARITY_USE_EOSPAC_

//WOULD THIS BE GOOD TO PARALLELIZE? KOKKOS/OPENMP?

//C++ Headers
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <filesystem>

//Data headers
#include <hdf5.h>
#include <hdf5_hl.h>

//Spiner headers
#include <ports-of-call/portability.hpp>
#include <singularity-eos/base/sp5/singularity_eos_sp5.hpp>
#include <spiner/databox.hpp>
#include <spiner/interpolation.hpp>
#include <spiner/sp5.hpp>
#include <spiner/spiner_types.hpp>
#include <ports-of-call/portability.hpp>

//Get EOS Models
#include <singularity-eos/eos/eos.hpp>
#include "singularity-eos/eos/eos_spiner.hpp"
#include <eospac-wrapper/eospac_wrapper.hpp>

using namespace singularity;
using namespace EospacWrapper;
using namespace std::chrono;
using duration = std::chrono::microseconds;
namespace fs = std::filesystem;
using DataBox = Spiner::DataBox<Real>;
using RegularGrid1D = Spiner::RegularGrid1D<Real>;
using Real = double;

// == Creates the Bounds of the Grid ==
class Bounds {
 public:
  Bounds(Real min, Real max, int N) : offset(0.0) {
    constexpr Real epsilon = std::numeric_limits<float>::epsilon();
    const Real min_offset = 10 * std::abs(epsilon);
    if (min <= 1e-16) offset = 1.1 * std::abs(min) + min_offset; //changed this from (min <= 0) because small values will make the log blow up
    min += offset;
    max += offset;
    min = std::log10(min);
    max = std::log10(max);
    dx = (max - min) / (N - 1);
    x0 = min;
    size = N;
  }
    Real i2lin(int i) const {
    return std::pow(10.0, x0 + static_cast<Real>(i) * dx) - offset; 
  }
 private:
  Real x0, dx, offset;
  int size; 
};

// == The following function creates and fills in a csv ==
void write_matrix_csv(const std::string& filename, const std::vector<std::vector<Real>>& matrix) {
            std::ofstream file(filename);
            file << std::setprecision(12);
            for (const auto& row : matrix) {
                for (size_t j = 0; j < row.size(); ++j) {
                    file << row[j];
                    if (j < row.size() - 1) file << ",";
                }
                file << "\n";
            }
        }


// == Main loop ===
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " path/to/output nRho nT sp5_file matid1 matid2 ..."; //Is this good usage? 
        return 1;
    }
    std::string base_output_path = "eos_benchmark_results/";
    base_output_path = argv[1];
    fs::path base_output_path(argv[1]);
    //if the folder does not exist, try to make one
    if (!fs::exists(base_output_path)) {
        if (!fs::create_directories(base_output_path)) {
            std::cerr << "Failed to create output directory: " << base_output_path << "\n";
            return 1; }}
    int nRho = std::stoi(argv[2]);  //test on batch sizes of 144, 269, 512, 2048 or round powers of ten
    int nT = std::stoi(argv[3]); 
    std::string sp5file = argv[4];

    std::vector<int> matids;
    for (int i = 5; i < argc; ++i)
        matids.push_back(std::atoi(argv[i]));

    // == Iterate through each material ==
    for (int matid : matids) {
        const int ntimes = 20; // Perform 20 time trials

        // == Get material metadata ==
        SesameMetadata meta;
        eosGetMetadata(matid, meta);

        // == Set up bounds ==
        Real rhoMin = 1.1 * std::max(meta.rhoMin, 1e-5);
        Real rhoMax = 0.9 * meta.rhoMax;
        Real TMin   = 1.1 * std::max(meta.TMin, 1.0);
        Real TMax   = 0.9 * meta.TMax;
        Bounds rhoBounds(rhoMin, rhoMax, nRho);
        Bounds TBounds(TMin, TMax, nT);

        std::vector<Real> rhos(nRho), temps(nT);
        for (int i = 0; i < nRho; ++i) rhos[i] = rhoBounds.i2lin(i);
        for (int j = 0; j < nT; ++j) temps[j] = TBounds.i2lin(j);

        // === Load EOS Models ===
        SpinerEOSDependsRhoT eos_rt(sp5file, matid);
        SpinerEOSDependsRhoSie eos_rs(sp5file, matid); //eventually make it SpinerEOSDependsRhoSie<NullTransfom>, for example
        EOSPAC eos_ref(matid); 

        //These vectors will store the compute time for each model for the 20 trials
        std::vector<double> time_e_eos_ref_list, time_e_rt_list, time_e_rs_list;
        std::vector<double> time_P_eos_ref_T_list, time_P_rt_T_list, time_P_rs_T_list;
        std::vector<double> time_P_eos_ref_sie_list, time_P_rt_sie_list; time_P_rs_sie_list;
        std::vector<double> time_T_back_eos_ref_list, time_T_back_rt_list, time_T_back_rs_list;

        //These matrices will be filled with the EOS values (overwriting for each trail)
        //Then at the end of the program, we export the matrix to a .csv to later use for accuracy tests
        Matrix<Real> e_eos_ref(nRho, nT), e_rt(nRho, nT), e_rs(nRho, nT);
        Matrix<Real> P_eos_ref_T(nRho, nT), P_rt_T(nRho, nT), P_rs_T(nRho, nT);
        Matrix<Real> P_eos_ref_sie(nRho, nT), P_rt_sie(nRho, nT), P_rs_sie(nRho, nT);
        Matrix<Real> T_back_eos_ref(nRho, nT), T_back_rt(nRho, nT), T_back_rs(nRho, nT);

        // === Benchmark Loop ===
        for (int rep = 0; rep < ntimes; ++rep) {

                // == EOSPAC (_eos) model ==
                auto t0 = high_resolution_clock::now(); //e(rho, T) 
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) { 
                        e_eos_ref(i, j) = eos_ref.InternalEnergyFromDensityTemperature(rhos[i], temps[j]); } }
                auto t1 = high_resolution_clock::now();
                time_eos_ref_e_list.push_back(duration<double, std::micro>(t1 - t0).count());

                t0 = high_resolution_clock::now(); // P(rho, T)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) {
                        P_eos_ref_T(i, j) = eos_ref.PressureFromDensityTemperature(rhos[i], temps[j]);}}
                t1 = high_resolution_clock::now();
                time_P_eos_ref_T_list.push_back(duration<double, std::micro>(t1 - t0).count());

                t0 = high_resolution_clock::now(); // P(rho, sie)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) { //is this line supposed to still be nT, it uses e_eos_ref(i,j)
                        P_eos_ref_sie(i, j) = eos_ref.PressureFromDensityInternalEnergy(rhos[i], e_eos_ref(i, j));}}
                t1 = high_resolution_clock::now();
                time_P_eos_ref_sie_list.push_back(duration<double, std::micro>(t1 - t0).count());

                t0 = high_resolution_clock::now(); //T(rho, sie)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) { //same with this one
                        T_back_eos_ref(i, j) = eos_ref.TemperatureFromDensityInternalEnergy(rhos[i], e_eos_ref(i, j));}}
                t1 = high_resolution_clock::now();
                time_T_back_eos_ref_list.push_back(duration<double, std::micro>(t1 - t0).count());



                // == SpinerEOSDependsRhoT (_rt) model ==
                t0 = high_resolution_clock::now(); //e(rho, T) 
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) {
                        e_rt(i, j) = rt.InternalEnergyFromDensityTemperature(rhos[i], temps[j]);}}
                t1 = high_resolution_clock::now();
                time_rt_e_list.push_back(duration<double, std::micro>(t1 - t0).count());

                t0 = high_resolution_clock::now(); // P(rho, T)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) {
                        P_rt_T(i, j) = rt.PressureFromDensityTemperature(rhos[i], temps[j]);}}
                t1 = high_resolution_clock::now();
                time_P_rt_T_list.push_back(duration<double, std::micro>(t1 - t0).count());

                t0 = high_resolution_clock::now(); // P(rho, sie)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) { //same with this one
                        P_rt_sie(i, j) = rt.PressureFromDensityInternalEnergy(rhos[i], e_rt(i, j));}}
                t1 = high_resolution_clock::now();
                time_P_rt_sie_list.push_back(duration<double, std::micro>(t1 - t0).count());

                t0 = high_resolution_clock::now(); //T(rho, sie)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) { //same with this one
                        T_back_rt(i, j) = rt.TemperatureFromDensityInternalEnergy(rhos[i], e_rt(i, j));}}
                t1 = high_resolution_clock::now();
                time_T_back_rt_list.push_back(duration<double, std::micro>(t1 - t0).count());



                // == SpinerEOSDependsRhoSie (_rs) model ==
                t0 = high_resolution_clock::now(); //e(rho, T) 
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) {
                        e_rs(i, j) = rs.InternalEnergyFromDensityTemperature(rhos[i], temps[j]);}}
                t1 = high_resolution_clock::now();
                time_rs_e_list.push_back(duration<double, std::micro>(t1 - t0).count());
                
                t0 = high_resolution_clock::now(); // P(rho, T)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) {
                        P_rs_T(i, j) = rs.PressureFromDensityTemperature(rhos[i], temps[j]);}}
                t1 = high_resolution_clock::now();
                time_P_rs_T_list.push_back(duration<double, std::micro>(t1 - t0).count());

                t0 = high_resolution_clock::now(); // P(rho, sie)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) { //same with this one
                        P_rs_sie(i, j) = rs.PressureFromDensityInternalEnergy(rhos[i], e_rs(i, j));}}
                t1 = high_resolution_clock::now();
                time_P_rs_sie_list.push_back(duration<double, std::micro>(t1 - t0).count());

                t0 = high_resolution_clock::now(); //T(rho, sie)
                for (int i = 0; i < nRho; ++i) {
                    for (int j = 0; j < nT; ++j) { //same with this one
                        T_back_rs(i, j) = rs.TemperatureFromDensityInternalEnergy(rhos[i], e_rs(i, j));}}
                t1 = high_resolution_clock::now();
                time_T_back_rs_list.push_back(duration<double, std::micro>(t1 - t0).count());

        } //end for loop for # of trials

    //Only save the filled in grid values on the last loop (only one grid exported to compare accuracies)
    write_matrix_csv((base_output_path / "eos_ref_internal_energy_" + matid + ".csv").string(), e_eos_ref); 
    write_matrix_csv((base_output_path / "rt_internal_energy_" + matid + ".csv").string(), e_rt);
    write_matrix_csv((base_output_path / "rs_internal_energy_" + matid + ".csv").string(), e_rs);
    write_matrix_csv((base_output_path / "T_back_eos_ref_" + matid + ".csv").string(), T_back_eos_ref);
    write_matrix_csv((base_output_path / "T_back_rt_" + matid + ".csv").string(), T_back_rt);
    write_matrix_csv((base_output_path / "T_back_rs_" + matid + ".csv").string(), T_back_rs);

    //after creating this .csv, should I average the last 15, can you explain again why the first 5 may be bad
    std::string timing_file = (base_output_path / "timing_" + matid + ".csv").string();
    std::ofstream timing_out(timing_file);
    timing_out << "eos_ref_e,rt_e,rs_e,P_eos_ref_T,P_rt_T,P_rs_T,P_eos_ref_sie,P_rt_sie,P_rs_sie,T_back_eos_ref,T_back_rt,T_back_rs\n";
    for (size_t i = 0; i < time_eos_ref_e_list.size(); ++i) {
        timing_out << time_eos_ref_e_list[i] << ","
                    << time_rt_e_list[i] << ","
                    << time_rs_e_list[i] << ","
                    << time_P_eos_ref_T_list[i] << ","
                    << time_P_rt_T_list[i] << ","
                    << time_P_rs_T_list[i] << ","
                    << time_P_eos_ref_sie_list[i] << ","
                    << time_P_rt_sie_list[i] << ","
                    << time_P_rs_sie_list[i] << ","
                    << time_T_back_eos_ref_list[i] << ","
                    << time_T_back_rt_list[i] << ","
                    << time_T_back_rs_list[i] << "\n";
    }

    std::cout << "Benchmark complete: " << timing_file << "\n";
    return 0;
}
}