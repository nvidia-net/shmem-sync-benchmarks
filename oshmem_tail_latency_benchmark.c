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

#define BENCHMARK "OpenSHMEM Sync Tail-Latency Test"
#define SKIP_DEFAULT                    (200)
#define ITERATIONS_DEFAULT              (100000)
#define MAX_PERCENTAGE_ARRAY_SIZE       (50)

typedef struct benchmark_func{
    void (*func_ptr)(void);
    char func_name[30];
}benchmark_func_t;

typedef struct data{
    double local, avg;
    double range_from, range_to;
}data_t;

void empty_func(){}

void swap(double *xp, double *yp)
{
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}
 
void bubble_sort(double arr[], int n)
{
   int i, j;
   for (i = 0; i < n-1; i++)     
       // Last i elements are already in place  
       for (j = 0; j < n-i-1; j++)
           if (arr[j] > arr[j+1])
              swap(&arr[j], &arr[j+1]);
}

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

void run_local_latencies_benchmark( void (*func)(void), int iterations, int skip, double* local_latencies, double *local_min, double *local_max, double* local_avg)
{
    double curr_latency;
    int i;
    *local_avg = 0;
    *local_min = __DBL_MAX__;
    *local_max = 0;
    for (i=0 ; i < (iterations + skip); i++)
    {
        int64_t t_start, t_stop;
        shmem_barrier_all();
        t_start = get_microsec_time_stamp();
        func();
        t_stop = get_microsec_time_stamp();
        curr_latency = t_stop - t_start;
        
        if (i >= skip) {
            local_latencies[i - skip] = curr_latency;
            *local_min = (*local_min < curr_latency) ? *local_min : curr_latency;
            *local_max = (*local_max > curr_latency) ? *local_max : curr_latency;
            *local_avg += curr_latency;
        }
    }
    *local_avg /= (double)iterations;
}

double percentile_latency(const double* arr, int arr_size, double percentage) 
{
    int index = (double)arr_size * percentage;
    if (index >= arr_size)
        index = arr_size - 1;
    return arr[index];
}

void print_results( FILE *stream, int my_pe, int iterations, int skip, int num_pes, double global_min, double global_max, 
                    data_t* avg, data_t* tails, double* percentages, int percentages_size, char* func_name)
{
    if (my_pe == 0) {
        int i;
        char temp_str[200];

        //Benchmark signature
        fprintf(stream, "# %s\n", BENCHMARK);

        //Results header
        fprintf(stream, "%*s", 22, "Noised-Avg");
        fprintf(stream, "%*s", 18, "Range");
        for(i = 0; i < percentages_size; i++)
            fprintf(stream, "%*.1f%%", 23, percentages[i] * 100.0);
        fprintf(stream, "%*s", 7, "Iter.");
        fprintf(stream, "%*s", 5, "skip");
        fprintf(stream, "%*s", 6, "#PEs");
        fprintf(stream, "%*s\n", 20, "Func.");

        //Results data
        sprintf(temp_str, "%.2f [%.2f-%.2f]", avg->avg, avg->range_from, avg->range_to);
        fprintf(stream, "%*s", 22, temp_str);
        sprintf(temp_str, "[%.2f-%.2f]", global_min, global_max);
        fprintf(stream, "%*s", 18, temp_str);
        for(i = 0; i < percentages_size; i++)
        {
            sprintf(temp_str, "%.2f [%.2f-%.2f]", tails[i].avg, tails[i].range_from, tails[i].range_to);
            fprintf(stream, "%*s", 24, temp_str);
        }
        fprintf(stream, "%*d", 7, iterations);
        fprintf(stream, "%*d", 5, skip);
        fprintf(stream, "%*d", 6, num_pes);
        fprintf(stream, "%*s\n", 20, func_name);
    }
}

void print_usage(FILE *stream, const char *prog, int my_pe)
{
    if (my_pe == 0)
    {
        fprintf(stream, " USAGE : %s [-i ITER] [-f FUNC] [-s SKIP] [-hv] [-V VERBOSE] [-p PERCENTAGE_LIST]\n", prog);
        fprintf(stream, "  -f : Select function {shmem_sync_all, shmem_barrier_all, sol} to benchmark.\n");
        fprintf(stream, "       By default, the value of FUNC is shmem_sync_all.\n");
        fprintf(stream, "  -i : Set number of iterations to ITER.\n");
        fprintf(stream, "       By default, the value of ITER is %d.\n", ITERATIONS_DEFAULT);
        fprintf(stream, "  -s : Set number of skip-iterations to SKIP.\n");
        fprintf(stream, "       By default, the value of SKIP is %d.\n", SKIP_DEFAULT);
        fprintf(stream, "  -p : List tail-latency percentages to measure.\n");
        fprintf(stream, "       By default, PERCENTAGE_LIST = {0.99, 0.95}.\n");
        fprintf(stream, "       e.g., -p 0.99,0.95\n");
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

int process_args(   FILE* stream, int argc, char *argv[], int my_pe, int *percentages_size, double *percentages,
                    int* iterations, int* skip, benchmark_func_t* f, int* verbosity_level)
{
    int c, i;
    char temp_str[200];
    char *temp_ptr;
    while ((c = getopt(argc, argv, ":hvi:s:f:V:p:")) != -1)
    {
        switch (c)
        {
        case 'f':
            if (strcmp(optarg, "shmem_sync_all") == 0) {
                f->func_ptr = &shmem_sync_all;
                strcpy(f->func_name, "shmem_sync_all");
            }
            else if (strcmp(optarg, "shmem_barrier_all") == 0) {
                f->func_ptr = &shmem_barrier_all;
                strcpy(f->func_name, "shmem_barrier_all");
            }
            else if (strcmp(optarg, "empty_func") == 0) {
                f->func_ptr = &empty_func;
                strcpy(f->func_name, "empty_func");
            }
            else {
                print_usage(stream, argv[0], my_pe);
                return 1;
            }
            break;
        
        case 'p':
            strcpy(temp_str, optarg);
            temp_ptr = strtok(temp_str, ",");
            for (i = 0 ; temp_ptr != NULL ; i++)
            {
                percentages[i] = atof(temp_ptr);
                if (percentages[i] < 0 || percentages[i] > 1)
                {
                    print_usage(stream, argv[0], my_pe);
                    return -1;
                }
                temp_ptr = strtok(NULL, ",");
            }
            *percentages_size = i;
            if(*percentages_size <= 0 || *percentages_size >= MAX_PERCENTAGE_ARRAY_SIZE){
                print_usage(stream, argv[0], my_pe);
                return -1;
            }
            break;

        case 'h':
            print_usage(stream, argv[0], my_pe);
            return 1;

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
    static double global_min, local_min, global_max, local_max;
    static data_t avg;
    static data_t tails[MAX_PERCENTAGE_ARRAY_SIZE];
    double percentages[MAX_PERCENTAGE_ARRAY_SIZE] = { 0.99, 0.95, 0 };
    int percentages_size = 2;
    double* local_latencies = NULL;
    int verbosity_level = 0, iterations = ITERATIONS_DEFAULT, skip = SKIP_DEFAULT;
    int my_pe, num_pes, i;
    FILE *stream = stdout;
    
    benchmark_func_t f;
    f.func_ptr = &shmem_sync_all;
    strcpy(f.func_name, "shmem_sync_all");
    
    for (i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; i += 1){
        pSyncRed1[i] = _SHMEM_SYNC_VALUE;
        pSyncRed2[i] = _SHMEM_SYNC_VALUE;
    }
        
    shmem_init();
    my_pe = shmem_my_pe();
    num_pes = shmem_n_pes();
    if (process_args(stream, argc, argv, my_pe, &percentages_size, percentages, &iterations, &skip, &f, &verbosity_level)){
        shmem_finalize();
        return EXIT_SUCCESS;
    }

    local_latencies = (double *)malloc(iterations * sizeof(double));
    if (!local_latencies)
    {
        fprintf(stream, "[%2d/%2d]: Allocation failed!\n", my_pe, num_pes);
        shmem_finalize();
        return EXIT_FAILURE;
    }
    
    run_local_latencies_benchmark(f.func_ptr, iterations, skip, local_latencies, &local_min, &local_max, &(avg.local));

    // Process Data...
    bubble_sort(local_latencies, iterations);

    for(i = 0; i < percentages_size; i++)
    {
        shmem_barrier_all();
        tails[i].local = percentile_latency(local_latencies, iterations, percentages[i]);
        shmem_double_min_to_all(&(tails[i].range_from), &(tails[i].local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
        shmem_double_max_to_all(&(tails[i].range_to), &(tails[i].local), 1, 0, 0, num_pes, pWrk1, pSyncRed1);
        shmem_double_sum_to_all(&(tails[i].avg), &(tails[i].local), 1, 0, 0, num_pes, pWrk2, pSyncRed2);
        tails[i].avg /= num_pes;
    }
    shmem_double_min_to_all(&global_min, &local_min         , 1, 0, 0, num_pes, pWrk1, pSyncRed1);
    shmem_double_max_to_all(&global_max, &local_max         , 1, 0, 0, num_pes, pWrk2, pSyncRed2);
    shmem_double_min_to_all(&(avg.range_from), &(avg.local) , 1, 0, 0, num_pes, pWrk1, pSyncRed1);
    shmem_double_max_to_all(&(avg.range_to), &(avg.local)   , 1, 0, 0, num_pes, pWrk2, pSyncRed2);
    shmem_double_sum_to_all(&(avg.avg), &(avg.local)        , 1, 0, 0, num_pes, pWrk1, pSyncRed1);
    avg.avg /= num_pes;
    

    print_results(stream, my_pe, iterations, skip, num_pes, global_min, global_max, 
                    &avg, tails, percentages, percentages_size, f.func_name);

    // For debugging...
    if (verbosity_level == 2) 
    {
        int i;
        for (i = 0; i < iterations; i++) {
            fprintf(stream, "[%4d:%4d]\titer %7d, latency %15.2f\n", my_pe, num_pes, i, local_latencies[i]);
        } 
    }

    free(local_latencies);
    shmem_finalize();
    return EXIT_SUCCESS;
}