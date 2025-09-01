// #ifndef _PAGESWAP_H_
// #define _PAGESWAP_H_
//
// #include "types.h"
// // #include "mmu.h"
//
// #define LIMIT 100
// extern int Th;       // Current threshold
// extern int Npg;      // Number of pages to swap out
// extern int alpha;    // α parameter
// extern int beta;     // β parameter
//
//
// void swap_out(void);
// void handle_pgfault(pde_t *pgdir, uint va);
// void invlpg(uint va);
// void adaptive_swap(void);
//
// #endif // _PAGESWAP_H_


#ifndef _PAGESWAP_H_
#define _PAGESWAP_H_

#include "types.h"
#include "mmu.h"

// Swap slot management
#define NSWAPSLOTS      800
#define BLOCKS_PER_PAGE 8

struct swap_slot {
    int page_perm;
    int is_free;
};

extern struct swap_slot swap_slots[NSWAPSLOTS];

// Adaptive parameters
#define LIMIT 100
extern int Th;       // Current threshold
extern int Npg;      // Pages to swap
extern int alpha;    // α parameter
extern int beta;     // β parameter

// Function prototypes
void swap_init(void);
int swap_alloc(int perm);
void swap_free(int slot_index);
void swap_write(int slot_index, void *page);
void swap_read(int slot_index, void *page);
void swap_out(void);
void handle_pgfault(pde_t *pgdir, uint va);
void invlpg(uint va);
void adaptive_swap(void);

#endif // _PAGESWAP_H_
