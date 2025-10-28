#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// --- Type Definitions and Global Constants ---

// Using 'int' for spins: +1 or -1
typedef int **Lattice;

// --- Helper Functions ---

// Function to generate a random float between 0.0 and 1.0
double rand_uniform() {
    return (double)rand() / (double)RAND_MAX;
}

// Allocates and initializes a 2D spin lattice (L x L)
Lattice allocate_lattice(int L) {
    Lattice spins = (Lattice)malloc(L * sizeof(int *));
    if (spins == NULL) {
        perror("Failed to allocate memory for rows");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < L; i++) {
        spins[i] = (int *)malloc(L * sizeof(int));
        if (spins[i] == NULL) {
            perror("Failed to allocate memory for columns");
            for (int k = 0; k < i; k++) free(spins[k]);
            free(spins);
            exit(EXIT_FAILURE);
        }
    }
    return spins;
}

// Frees the memory allocated for a 2D spin lattice
void free_lattice(Lattice spins, int L) {
    for (int i = 0; i < L; i++) {
        free(spins[i]);
    }
    free(spins);
}

// Copies the source lattice to the destination lattice
void copy_lattice(Lattice dest, Lattice src, int L) {
    for (int i = 0; i < L; i++) {
        memcpy(dest[i], src[i], L * sizeof(int));
    }
}

// Copies an array of doubles
void copy_array(double *dest, const double *src, int size) {
    memcpy(dest, src, size * sizeof(double));
}

// --- Ising Model Core Functions ---

// Initializes a 2D spin lattice (L x L) randomly to +1 or -1
void initial_lattice(Lattice spins, int L) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            // 2 * (rand_uniform() > 0.5) - 1
            spins[i][j] = (rand_uniform() > 0.5) ? 1 : -1;
        }
    }
}

// Applies periodic boundary conditions
int periodic(int i, int L) {
    // C's % operator can return negative for negative i, so a safer approach is:
    return (i % L + L) % L;
}

// Calculates the change in energy if spin (i, j) is flipped
int delta_energy(Lattice spins, int L, int i, int j) {
    int S = spins[i][j];
    int nb = (
        spins[periodic(i + 1, L)][j] +
        spins[periodic(i - 1, L)][j] +
        spins[i][periodic(j + 1, L)] +
        spins[i][periodic(j - 1, L)]
    );
    return 2 * S * nb;
}

// Calculates the total energy of the lattice
int total_energy(Lattice spins, int L) {
    int E = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int S = spins[i][j];
            // Only count neighbors to the right and down to avoid double-counting interactions
            int nb = spins[periodic(i + 1, L)][j] + spins[i][periodic(j + 1, L)];
            E -= S * nb; // H = -J * sum(Si*Sj), J=1
        }
    }
    return E;
}

// Performs one Monte Carlo sweep using the Metropolis algorithm
void metropolis(Lattice spins, int L, double T) {
    for (int k = 0; k < L * L; k++) {
        // Pick a random spin
        int i = (int)(rand_uniform() * L);
        int j = (int)(rand_uniform() * L);

        int dE = delta_energy(spins, L, i, j);

        if (dE <= 0 || rand_uniform() < exp(-dE / T)) {
            spins[i][j] *= -1; // Flip the spin
        }
    }
}

// Calculates the total magnetization of the lattice
int calc_magnetization(Lattice spins, int L) {
    int M = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            M += spins[i][j];
        }
    }
    return M;
}

// Performs a replica exchange attempt between neighboring replicas i and i+1
void replica_exchange(Lattice *replicas, int num_replicas, int L, double *energies, const double *betas) {
    for (int i = 0; i < num_replicas - 1; i++) {
        double beta_i = betas[i];
        double beta_j = betas[i + 1];
        double E_i = energies[i];
        double E_j = energies[i + 1];

        // Acceptance criterion: A(i -> j) = min(1, exp(d)) where d = (beta_j - beta_i) * (E_i - E_j)
        double d = (beta_j - beta_i) * (E_i - E_j);

        if (d <= 0 || rand_uniform() < exp(-d)) {
            // Swap lattices (pointers/references in Python, content swap in C)
            // C: we must swap the actual lattice contents and the energy values.

            // 1. Swap Energies
            double tmp_E = energies[i];
            energies[i] = energies[i + 1];
            energies[i + 1] = tmp_E;

            // 2. Swap Lattices (using a temporary copy)
            Lattice tmp_lattice = allocate_lattice(L);
            copy_lattice(tmp_lattice, replicas[i], L);
            copy_lattice(replicas[i], replicas[i + 1], L);
            copy_lattice(replicas[i + 1], tmp_lattice, L);
            free_lattice(tmp_lattice, L);
        }
    }
}

// Calculates the average and error (std dev of mean)
void calc_ave_err(const double *ms, int step_expect, double *ave, double *err) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < step_expect; i++) {
        sum += ms[i];
        sum_sq += ms[i] * ms[i];
    }
    *ave = sum / step_expect;

    // Standard deviation / sqrt(N)
    double variance = (sum_sq / step_expect) - (*ave) * (*ave);
    if (variance < 0.0) variance = 0.0; // Avoid issues with floating point precision
    *err = sqrt(variance) / sqrt(step_expect);
}

// --- Main Monte Carlo Simulation ---

void monte_carlo(int L, int step_thermal, int step_expect, int exchange_interval, const double *temps, int num_replicas) {
    
    // Calculate betas
    double *betas = (double *)malloc(num_replicas * sizeof(double));
    if (betas == NULL) { perror("malloc failed for betas"); exit(EXIT_FAILURE); }
    for (int i = 0; i < num_replicas; i++) {
        betas[i] = 1.0 / temps[i];
    }

    // Initialization: replicas and energies
    Lattice *replicas = (Lattice *)malloc(num_replicas * sizeof(Lattice));
    if (replicas == NULL) { perror("malloc failed for replicas"); exit(EXIT_FAILURE); }
    double *energies = (double *)malloc(num_replicas * sizeof(double));
    if (energies == NULL) { perror("malloc failed for energies"); exit(EXIT_FAILURE); }

    for (int i = 0; i < num_replicas; i++) {
        replicas[i] = allocate_lattice(L);
        initial_lattice(replicas[i], L);
        energies[i] = (double)total_energy(replicas[i], L);
    }

    printf("--- Thermalization (%d steps) ---\n", step_thermal);
    // Thermalization
    for (int step = 0; step < step_thermal; step++) {
        for (int i = 0; i < num_replicas; i++) {
            metropolis(replicas[i], L, temps[i]);
            energies[i] = (double)total_energy(replicas[i], L);
        }
        if (step % exchange_interval == 0) {
            replica_exchange(replicas, num_replicas, L, energies, betas);
        }
        if (step % 1000 == 0 && step > 0) {
            printf("Thermalization Step %d/%d done.\n", step, step_thermal);
        }
    }

    printf("--- Collecting Data (%d steps) ---\n", step_expect);
    // Collect data
    // Data arrays are indexed [replica_index][step]
    double **m1s = (double **)malloc(num_replicas * sizeof(double *));
    double **m2s = (double **)malloc(num_replicas * sizeof(double *));
    double **m4s = (double **)malloc(num_replicas * sizeof(double *));
    double **e1s = (double **)malloc(num_replicas * sizeof(double *));
    double **e2s = (double **)malloc(num_replicas * sizeof(double *));

    for (int i = 0; i < num_replicas; i++) {
        m1s[i] = (double *)calloc(step_expect, sizeof(double));
        m2s[i] = (double *)calloc(step_expect, sizeof(double));
        m4s[i] = (double *)calloc(step_expect, sizeof(double));
        e1s[i] = (double *)calloc(step_expect, sizeof(double));
        e2s[i] = (double *)calloc(step_expect, sizeof(double));
    }

    for (int step = 0; step < step_expect; step++) {
        for (int i = 0; i < num_replicas; i++) {
            metropolis(replicas[i], L, temps[i]);
            energies[i] = (double)total_energy(replicas[i], L);
            
            // Collect data
            double M = (double)calc_magnetization(replicas[i], L);
            double E = energies[i];

            e1s[i][step] = E;
            e2s[i][step] = E * E;
            m1s[i][step] = M;
            m2s[i][step] = M * M;
            m4s[i][step] = M * M * M * M;
        }
        if (step % exchange_interval == 0) {
            replica_exchange(replicas, num_replicas, L, energies, betas);
        }
        if (step % 1000 == 0) {
            printf("Step %d/%d done.\n", step, step_expect);
        }
    }

    // --- Process and Save Data ---
    
    // Save data to a file (only for the main part for simplicity)
    // The Python code saves per-seed and then the final average.
    // This C code only implements the per-seed calculation and output.
    
    char filename[100];
    // In a final setup, the seed or other unique ID would be here
    snprintf(filename, sizeof(filename), "dat_2d_ising_L%d_seed_single.txt", L); 
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Failed to open file for output");
        // Clean up and exit
        // ... (cleanup code)
        return;
    }

    fprintf(fp, "T, am1a, am1e, m1a, m1e, m2a, m2e, m4a, m4e, e1a, e1e, e2a, e2e, susc, spec, bind\n");

    double L_sq = (double)L * L;
    double L_pow4 = L_sq * L_sq;
    double L_pow8 = L_pow4 * L_pow4;
    
    for (int i = 0; i < num_replicas; i++) {
        double am1s_abs[step_expect]; // For |M|
        for (int k=0; k<step_expect; k++) am1s_abs[k] = fabs(m1s[i][k]);

        // Calculate Averages and Errors
        double am1a, am1e, m1a, m1e, m2a, m2e, m4a, m4e, e1a, e1e, e2a, e2e;
        
        // M1 (|M|/L^2)
        calc_ave_err(am1s_abs, step_expect, &am1a, &am1e);
        am1a /= L_sq; am1e /= L_sq;
        // M1 (M/L^2)
        calc_ave_err(m1s[i], step_expect, &m1a, &m1e);
        m1a /= L_sq; m1e /= L_sq;
        // M2 (M^2/L^4)
        calc_ave_err(m2s[i], step_expect, &m2a, &m2e);
        m2a /= L_pow4; m2e /= L_pow4;
        // M4 (M^4/L^8)
        calc_ave_err(m4s[i], step_expect, &m4a, &m4e);
        m4a /= L_pow8; m4e /= L_pow8;
        // E1 (E/L^2)
        calc_ave_err(e1s[i], step_expect, &e1a, &e1e);
        e1a /= L_sq; e1e /= L_sq;
        // E2 (E^2/L^4)
        calc_ave_err(e2s[i], step_expect, &e2a, &e2e);
        e2a /= L_pow4; e2e /= L_pow4;

        // Calculate Susceptibility, Specific Heat, and Binder Cumulant
        double temp = temps[i];
        double susc = (m2a - am1a * am1a) / temp * L_sq;
        double spec = (e2a - e1a * e1a) / (temp * temp) * L_sq;
        
        // Binder Cumulant: B = 1.0 - <M^4> / (3 * <M^2>^2)
        double bind;
        if (m2a == 0.0) {
            bind = 1.0 - m4a / (1e-16 * 3.0);
        } else {
            bind = 1.0 - m4a / (m2a * m2a * 3.0);
        }
        
        // Output result for this replica's temperature
        fprintf(fp, "%+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f\n",
                temp, am1a, am1e, m1a, m1e, m2a, m2e, m4a, m4e, e1a, e1e, e2a, e2e, susc, spec, bind);
    }
    
    fclose(fp);
    printf("Results saved to %s\n", filename);


    // --- Cleanup ---
    for (int i = 0; i < num_replicas; i++) {
        free_lattice(replicas[i], L);
        free(m1s[i]); free(m2s[i]); free(m4s[i]); 
        free(e1s[i]); free(e2s[i]);
    }
    free(replicas);
    free(energies);
    free(betas);
    free(m1s); free(m2s); free(m4s);
    free(e1s); free(e2s);
}


// --- Main Function with Argument Parsing (Simplified) ---

void main() {
    // Simplified argument parsing: hardcode defaults or read from console
//    int L = 16;
    int L = 64;
    int step_thermal = 5000;
    int step_expect = 20000;
    int exchange_interval = 1;

    // Hardcoded temperature range (0.5 to 5.0 in steps of 0.1)
    // The Python code runs this over a loop of seeds and averages.
    // This C version is simplified to run only once.
    double temps_start = 0.5;
    double temps_end = 5.01;
    double temps_step = 0.1;

    int num_replicas = (int)ceil((temps_end - temps_start) / temps_step);
    double *temps = (double *)malloc(num_replicas * sizeof(double));
    if (temps == NULL) { perror("malloc failed for temps"); exit(EXIT_FAILURE); }
    for (int i = 0; i < num_replicas; i++) {
        temps[i] = temps_start + i * temps_step;
    }

    // Initialize random seed
    // Using time(NULL) for a single run, or you could use a fixed seed (e.g., 12345)
    srand(12345); // Fixed seed for reproducibility

    printf("2D Ising Replica Exchange Monte Carlo Simulation\n");
    printf("L: %d, Thermal: %d, Expect: %d, Replicas: %d\n", L, step_thermal, step_expect, num_replicas);
    
    monte_carlo(L, step_thermal, step_expect, exchange_interval, temps, num_replicas);

    free(temps);
    printf("Simulation finished.\n");
}
