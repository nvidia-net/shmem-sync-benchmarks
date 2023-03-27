#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include <shmem.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#define BENCHMARK                       "OpenSHMEM overlap benchmark for sync operation"
#define SKIP_DEFAULT                    (200)
#define ITERATIONS_DEFAULT              (100000)
#define KB                              (1024)
#define MB                              (1024*KB)
#define COMPUTE_BUFFER_SIZE             (512)


// SHMEM API version 1.4 doesn't support non-blocking sync operation!
// Thus, I had to add it by myself...
void shmem_sync_all_post(void);
void shmem_sync_all_wait(void);


typedef struct data{
    double local, avg;
    double range_from, range_to;
}data_t;

int64_t get_microsec_time_stamp()
{
    int64_t retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL))
    {
        perror("gettimeofday");
        abort();
    }
    retval = ((int64_t)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}

void swap(volatile double *volatile xp, volatile double *volatile yp)
{
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}
 
void computation_func(volatile double *volatile computation_arr, int computation_amount)
{
    //Bubble sort
    int i, j;
    for (i = 0; i < computation_amount-1; i++)     
       // Last i elements are already in place  
       for (j = 0; j < computation_amount-i-1; j++)
           if (computation_arr[j] > computation_arr[j+1])
              swap(&computation_arr[j], &computation_arr[j+1]);
}

double computation_latency(volatile double *volatile computation_arr, int computation_amount, int iterations, int skip)
{
    int64_t t_start, t_stop;
    int i;
    for (i = 0; i < skip; i++)
        computation_func(computation_arr, computation_amount);
    t_start = get_microsec_time_stamp();
    for (i = 0; i < iterations; i++)
        computation_func(computation_arr, computation_amount);
    t_stop = get_microsec_time_stamp();
    return (double)(t_stop - t_start) / (double)iterations;
}

double computation_and_networking_latency(volatile double *volatile computation_arr, int computation_amount, int iterations, int skip)
{
    int64_t t_start, t_stop;
    int i;
    for (i = 0; i < skip; i++)
    {
        shmem_sync_all_post();
        computation_func(computation_arr, computation_amount);
        shmem_sync_all_wait();
    }    
    shmem_barrier_all();
    t_start = get_microsec_time_stamp();
    for (i = 0; i < iterations; i++)
    {
        shmem_sync_all_post();
        computation_func(computation_arr, computation_amount);
        shmem_sync_all_wait();
    }
    t_stop = get_microsec_time_stamp();
    return (double)(t_stop - t_start) / (double)iterations;
}

void print_usage(FILE *stream, const char *prog, int my_pe)
{
    if (my_pe == 0)
    {
        fprintf(stream, " USAGE : %s [-i ITER] [-s SKIP] [-hv] [-V VERBOSE]\n", prog);
        fprintf(stream, "  -i : Set number of iterations to ITER.\n");
        fprintf(stream, "       By default, the value of ITER is %d.\n", ITERATIONS_DEFAULT);
        fprintf(stream, "  -s : Set number of skip-iterations to SKIP.\n");
        fprintf(stream, "       By default, the value of SKIP is %d.\n", SKIP_DEFAULT);
        fprintf(stream, "  -h : Print this help.\n");
        fprintf(stream, "  -v : Print version info.\n");
        fprintf(stream, "  -V : Set verbosity level {0=low, 1, 2=high}.\n");
        fprintf(stream, "       By default, the value of VERBOSE is 0.\n");
        fprintf(stream, "\n");
        fflush(stream);
    }
}

void print_version(FILE *stream, int my_pe)
{
    if (my_pe == 0) {
        int major, minor;
        char name[SHMEM_MAX_NAME_LEN];
        shmem_info_get_version(&major, &minor);
        shmem_info_get_name(name);
        fprintf(stream, "# %s\n", BENCHMARK);
        fprintf(stream, "# Implementation version (API) %d.%d\n", major, minor);
        fprintf(stream, "# vendor defined name: %s\n", name);
        fflush(stream);
    }
}

int process_args(FILE* stream, int argc, char *argv[], int my_pe, int* iterations, int* skip, int* verbosity_level)
{
    int c;
    while ((c = getopt(argc, argv, ":vi:s:V:")) != -1)
    {
        switch (c)
        {
        case 'v':
            print_version(stream, my_pe);
            return 1;

        case 'V':
            *verbosity_level = atoi(optarg);
            if (*verbosity_level < 0 || *verbosity_level > 2)
            {
                print_usage(stream, argv[0], my_pe);
                return -1;
            }
            break;

        case 's':
            *skip = atoi(optarg);
            if (*skip < 0)
            {
                print_usage(stream, argv[0], my_pe);
                return -1;
            }
            break;

        case 'i':
            *iterations = atoi(optarg);
            if (*iterations < 1)
            {
                print_usage(stream, argv[0], my_pe);
                return -1;
            }
            break;
        
        default:
            print_usage(stream, argv[0], my_pe);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    static long pSyncRed1[_SHMEM_REDUCE_SYNC_SIZE];
    static long pSyncRed2[_SHMEM_REDUCE_SYNC_SIZE];
    static double pWrk1[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
    static double pWrk2[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
    static data_t compute, network, overall, overhead, availability;
    volatile double *volatile computation_arr;
    int verbosity_level = 0, iterations = ITERATIONS_DEFAULT, skip = SKIP_DEFAULT;
    int my_pe, num_pes, i;
    FILE *stream = stdout;
    
    for (i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; i += 1){
        pSyncRed1[i] = _SHMEM_SYNC_VALUE;
        pSyncRed2[i] = _SHMEM_SYNC_VALUE;
    }
    shmem_init();
    my_pe = shmem_my_pe();
    num_pes = shmem_n_pes();

    if (process_args(stream, argc, argv, my_pe, &iterations, &skip, &verbosity_level) != 0)
    {
        shmem_finalize();
        return 0;
    }        

    if (my_pe == 0)
    {
        fprintf(stream, "%*s   ", 18, "Computation-Amount");
        fprintf(stream, "%*s   ", 24, "Overall-Latency");
        fprintf(stream, "%*s   ", 24, "Network-latency");
        fprintf(stream, "%*s   ", 24, "Computation-Latency");
        fprintf(stream, "%*s   ", 24, "Overhead");
        fprintf(stream, "%*s\n", 24, "Availability");
    }

    computation_arr = (double *) malloc(COMPUTE_BUFFER_SIZE * sizeof(double));
    if(computation_arr == NULL){
        fprintf(stream, "Allocation Failed!\n");
        shmem_finalize();
    }
    
    network.local = computation_and_networking_latency(NULL, 0, iterations, skip);
    shmem_double_min_to_all(&(network.range_from)   , &(network.local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
    shmem_double_max_to_all(&(network.range_to)     , &(network.local), 1, 0, 0, num_pes, pWrk1, pSyncRed1);
    shmem_double_sum_to_all(&(network.avg)          , &(network.local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
    network.avg /= num_pes;

    for (i = 1 ;i < (COMPUTE_BUFFER_SIZE+1) ;i*=2)
    {
        compute.local  = computation_latency               (computation_arr, i, iterations, skip);
        overall.local  = computation_and_networking_latency(computation_arr, i, iterations, skip);
        overhead.local = overall.local - compute.local;
        availability.local = 1 - (overhead.local / network.local);

        shmem_barrier_all();
        shmem_double_min_to_all(&(compute.range_from)   , &(compute.local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
        shmem_double_max_to_all(&(compute.range_to)     , &(compute.local), 1, 0, 0, num_pes, pWrk1, pSyncRed1);
        shmem_double_sum_to_all(&(compute.avg)          , &(compute.local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
        compute.avg /= num_pes;
        shmem_double_min_to_all(&(overall.range_from)   , &(overall.local), 1, 0, 0, num_pes, pWrk1, pSyncRed1);
        shmem_double_max_to_all(&(overall.range_to)     , &(overall.local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
        shmem_double_sum_to_all(&(overall.avg)          , &(overall.local), 1, 0, 0, num_pes, pWrk1, pSyncRed1);
        overall.avg /= num_pes;
        shmem_double_min_to_all(&(overhead.range_from)   , &(overhead.local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
        shmem_double_max_to_all(&(overhead.range_to)     , &(overhead.local), 1, 0, 0, num_pes, pWrk1, pSyncRed1);
        shmem_double_sum_to_all(&(overhead.avg)          , &(overhead.local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
        overhead.avg /= num_pes;
        shmem_double_min_to_all(&(availability.range_from)   , &(availability.local), 1, 0, 0, num_pes, pWrk1, pSyncRed1);
        shmem_double_max_to_all(&(availability.range_to)     , &(availability.local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
        shmem_double_sum_to_all(&(availability.avg)          , &(availability.local), 1, 0, 0, num_pes, pWrk1, pSyncRed1);
        availability.avg /= num_pes;

        if (my_pe == 0)
        {
            char temp_str[200];
            fprintf(stream, "%18d   ", i);
            sprintf(temp_str, "%.2f [%.2f-%.2f]", overall.avg, overall.range_from, overall.range_to);
            fprintf(stream, "%*s   ", 24, temp_str);
            sprintf(temp_str, "%.2f [%.2f-%.2f]", network.avg, network.range_from, network.range_to);
            fprintf(stream, "%*s   ", 24, temp_str);
            sprintf(temp_str, "%.2f [%.2f-%.2f]", compute.avg, compute.range_from, compute.range_to);
            fprintf(stream, "%*s   ", 24, temp_str);
            sprintf(temp_str, "%.2f [%.2f-%.2f]", overhead.avg, overhead.range_from, overhead.range_to);
            fprintf(stream, "%*s   ", 24, temp_str);
            sprintf(temp_str, "%.2f [%.2f-%.2f]", availability.avg, availability.range_from, availability.range_to);
            fprintf(stream, "%*s\n", 24, temp_str);
        } 
    }
    if (my_pe == 0) {
        char temp_str[200];
        //Benchmark signature
        fprintf(stream, "# %s\n", BENCHMARK);
        //Results header
        fprintf(stream, "%*s", 7, "Iter.");
        fprintf(stream, "%*s", 5, "skip");
        fprintf(stream, "%*s\n", 6, "#PEs");
        //Results data
        fprintf(stream, "%*d", 7, iterations);
        fprintf(stream, "%*d", 5, skip);
        fprintf(stream, "%*d\n", 6, num_pes);
    }
    
    free((void *)computation_arr);
    shmem_finalize();
    return 0;
}

