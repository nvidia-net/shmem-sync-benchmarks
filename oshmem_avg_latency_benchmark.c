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

#define BENCHMARK "OpenSHMEM shmem_sunc_all() avg latency Test"
#define SKIP_DEFAULT                    (200)
#define ITERATIONS_DEFAULT              (10000)

long pSyncRed1[_SHMEM_REDUCE_SYNC_SIZE];
long pSyncRed2[_SHMEM_REDUCE_SYNC_SIZE];
double pWrk1[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
double pWrk2[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];

void empty_func(){}

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

void run_local_avg_latency_benchmark(void (*func)(void), int iterations, int skip, double* local_avg)
{
    int64_t t_start, t_stop;
    int i = 0;
    for (i = 0; i < skip; i++)
        func();
    shmem_barrier_all();
    t_start = get_microsec_time_stamp();
    for (i = 0; i < iterations; i++)
        func();
    t_stop = get_microsec_time_stamp();
    *local_avg =  (t_stop - t_start) / (double)iterations;
}

void print_results(FILE *stream, int verbosity_level, int my_pe, int iterations, int skip, int num_pes,
                    double* global_avg, double* global_min, double* global_max, char* func_name)
{
    shmem_barrier_all();
    if (my_pe == 0) {
        fprintf(stream, "# %s\n", BENCHMARK);
        
        //Results header
        fprintf(stream, "%*s", 5, "# Avg");
        fprintf(stream, "%*s", 10, "Min");
        fprintf(stream, "%*s", 10, "Max");
        fprintf(stream, "%*s", 7, "Iter.");
        fprintf(stream, "%*s", 5, "skip");
        fprintf(stream, "%*s", 6, "#PEs");
        fprintf(stream, "%*s\n", 20, "Func.");
        fflush(stream);

        //Results data
        fprintf(stream, "%*.2f", 5, *global_avg);
        fprintf(stream, "%*.2f", 10, *global_min);
        fprintf(stream, "%*.2f", 10, *global_max);
        fprintf(stream, "%*d", 7, iterations);
        fprintf(stream, "%*d", 5, skip);
        fprintf(stream, "%*d", 6, num_pes);
        fprintf(stream, "%*s\n", 20, func_name);
    }
    
    if (verbosity_level == 1)
    {
        const int buffer_size = 300;
        char local_buff[buffer_size];
        char* remote_buff = shmem_malloc(buffer_size);
        char hostname[buffer_size];
        gethostname(hostname, sizeof(hostname));
        sprintf(remote_buff, "[%4d/%4d]: PID %d, Host %s, skip %d, iter %d\n", my_pe, num_pes, getpid(), hostname, skip, iterations);
        
        if(my_pe == 0){
            int pe;
            for(pe=0; pe < num_pes; pe++){
                shmem_char_get(local_buff, remote_buff, buffer_size, pe);
                fprintf(stream, local_buff);
            }
        }
        shmem_free(remote_buff);
    }    
}

void print_usage(FILE *stream, const char *prog, int my_pe)
{
    if (my_pe == 0)
    {
        fprintf(stream, " USAGE : %s [-i ITER] [-f FUNC] [-s SKIP] [-hv] [-V VERBOSE]\n", prog);
        fprintf(stream, "  -f : Select function {shmem_sync_all, shmem_barrier_all, empty_func} to benchmark.\n");
        fprintf(stream, "       By default, the value of FUNC is shmem_sync_all.\n");
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

int process_args(FILE* stream, int argc, char *argv[], int my_pe, int* iterations, int* skip, void (**func_ptr)(void), char* func_name, int* verbosity_level)
{
    int c;
    while ((c = getopt(argc, argv, ":vi:s:f:V:")) != -1)
    {
        switch (c)
        {
        case 'f':
            if (strcmp(optarg, "shmem_sync_all") == 0) {
                *func_ptr = &shmem_sync_all;
                strcpy(func_name, "shmem_sync_all");
            }
            else if (strcmp(optarg, "shmem_barrier_all") == 0) {
                *func_ptr = &shmem_barrier_all;
                strcpy(func_name, "shmem_barrier_all");
            }
            else if (strcmp(optarg, "empty_func") == 0) {
                *func_ptr = &empty_func;
                strcpy(func_name, "empty_func");
            }
            else {
                print_usage(stream, argv[0], my_pe);
                return 1;
            }
            break;

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
    static double min_avg, max_avg;
    static double global_avg = 0, local_avg=0;
    int verbosity_level = 0, iterations = ITERATIONS_DEFAULT, skip = SKIP_DEFAULT;
    int my_pe, num_pes, i;
    FILE *stream = stdout;
    void (*func_ptr)(void) = &shmem_sync_all;
    char func_name[30] = "shmem_sync_all";
    
    for (i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; i += 1){
        pSyncRed1[i] = _SHMEM_SYNC_VALUE;
        pSyncRed2[i] = _SHMEM_SYNC_VALUE;
    }
    shmem_init();
    my_pe = shmem_my_pe();
    num_pes = shmem_n_pes();

    if (process_args(stream, argc, argv, my_pe, &iterations, &skip, &func_ptr, func_name, &verbosity_level) != 0){
        shmem_finalize();
        return 0;
    }        
    run_local_avg_latency_benchmark(func_ptr, iterations, skip, &local_avg);

    shmem_barrier_all();
    shmem_double_min_to_all(&min_avg,    &local_avg, 1, 0, 0, num_pes, pWrk1, pSyncRed1);
    shmem_double_max_to_all(&max_avg,    &local_avg, 1, 0, 0, num_pes, pWrk2, pSyncRed2);
    shmem_double_sum_to_all(&global_avg, &local_avg, 1, 0, 0, num_pes, pWrk1, pSyncRed1);
    global_avg /= num_pes;

    print_results(stream, verbosity_level, my_pe, iterations, skip, num_pes, &global_avg, &min_avg, &max_avg, func_name);

    shmem_finalize();
    return 0;
}

