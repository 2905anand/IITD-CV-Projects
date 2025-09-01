#include "types.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "mmu.h"
#include "proc.h"
#include "spinlock.h"
#include "sleeplock.h"
#include "fs.h"
#include "buf.h"
#include "pageswap.h"


extern struct {
  struct spinlock lock;
  struct proc proc[NPROC];
} ptable;



struct swap_slot swap_slots[NSWAPSLOTS];
int alpha = ALPHA;
int beta = BETA;
int Th = 100;
int Npg = 4;


void invlpg(uint va) {
  asm volatile("invlpg (%0)" : : "r" (va) : "memory");
}


int incr(int i, int val){
  i = i+val;
  return i;
}


uint getBlockIndex(int index){
  int value = 2 + index * BLOCKS_PER_PAGE;
  return value;
}

void switching_functions(int value){
  switch(value){
  case 1: 
     {
        int i = 0;
        while(i < NSWAPSLOTS){
          assign(1, 0, i);
          i = incr(i,1);
        }
     }
     break;
  }
}


int swap_alloc(int perm) {
  int i=0;
  while(i < NSWAPSLOTS){
    if(swap_slots[i].is_free == 0){
    }else{
      assign(0, perm, i);
      return i;
    }
    i = incr(i, 1);
  }
  return -1;
}

void assign(int is_free, int page_perm, int index){
    swap_slots[index].is_free = is_free;
    swap_slots[index].page_perm = page_perm;
}

static struct proc* find_victim_process(void) {
  struct proc *p, *victim = 0;
  int max_rss_ = -1;

  p = ptable.proc;
  while(p < &ptable.proc[NPROC]){
    if(p->state != UNUSED && p->rss == max_rss_) {
      if(victim && p->pid >= victim->pid){
      }else if(victim && p->pid < victim->pid) {
        victim = p;
      }
    }else if(p->state != UNUSED && p->rss > max_rss_) {
      max_rss_ = p->rss;
      victim = p;
    }
    p++;
  }
  return victim;
}


void swap_free(int slot_index) {
  if (slot_index < 0 || slot_index >= NSWAPSLOTS || slot_index == -1 * PGSIZE)
    panic("swap_free: invalid slot");
  assign(1, 0, slot_index);
}

void swap_init(void) {
  int i = 0;
  switching_functions(1);
  cprintf("Swap: Initialized %d swap slots\n", NSWAPSLOTS);
}


static pte_t* find_victim_page(struct proc *p) {
  pde_t *pgdir = p->pgdir;
  pte_t *pte, *fallback = 0;
  uint va;
  va = 0;

  while( va < KERNBASE){
    pte = walkpgdir(pgdir, (void*)va, 0);
    if(pte && !(*pte & PTE_P)){
    }else if(pte && (*pte & PTE_P)) {             
      if((*pte & PTE_A)){
        if(!fallback) fallback = pte;      
        *pte &= ~PTE_A; 
      }else{
        return pte;
      }
    }
    va = incr(va, PGSIZE);
  }
  return fallback; // Fallback to first accessed page
}



void swap_write(int slot_index, void *page) {
  uint start_block = getBlockIndex(slot_index);
  int i = 0;
  while(i  < BLOCKS_PER_PAGE){
    struct buf *b = bread(0, start_block + i);
    memmove(b->data, (char*)page + i*BSIZE, BSIZE);
    bwrite(b);
    brelse(b);
    i = incr(i,1);
  }
}


void swap_out(void) {
  struct proc *victim;
  pte_t *pte;
  uint pa, perm, slot;
  char *mem;
  victim = find_victim_process();
  if(victim == 0)
    panic("swap_out: no victim process");
  pte = find_victim_page(victim);
  if(pte == 0){
    panic("swap_out: no victim page");
  }

  pa = PTE_ADDR(*pte);
  perm = (*pte & (PTE_U | PTE_W));

  slot = swap_alloc(perm);
  if(slot < 0){
    panic("swap_out: no swap slots");
  }
  mem = P2V(pa);
  swap_write(slot, mem);
  *pte = (slot << 12) | PTE_SWAP;
  kfree(mem);
  int pg_s = PGSIZE;
  victim->ms -= pg_s;
  victim->rss--;
}


void swap_read(int slot_index, void *page) {
  
  uint start_block = getBlockIndex(slot_index);
  int i = 0;
  while(i < BLOCKS_PER_PAGE){
    struct buf *b = bread(0, start_block + i);
    memmove((char*)page + i*BSIZE, b->data, BSIZE);
    brelse(b);
    i = incr(i, 1);
  }
}

void handle_pgfault(pde_t *pgdir, uint va) {
  pte_t *pte = walkpgdir(pgdir, (void*)va, 0);
  if(pte){
  }else if(!pte || !(*pte & PTE_SWAP))
    panic("handle_pgfault: invalid page fault");
  int shift  = 12;
  int slot = (PTE_ADDR(*pte)) >> shift;
  char *mem = kalloc();
  if(mem){
  }else{
    swap_out();
    mem = kalloc();
    if(!mem) panic("handle_pgfault: kalloc failed");
  }
  swap_read(slot, mem);
  *pte = V2P(mem) | PTE_P | swap_slots[slot].page_perm;
  swap_free(slot);
  int pg_size = PGSIZE;
  struct proc *curproc = myproc();
  curproc->ms = curproc->ms + pg_size;
  curproc->rss = curproc->rss + 1;
}

extern struct {
  struct spinlock lock;
  int use_lock;
  struct run *freelist;
  int free_pages;
} kmem;

void adaptive_swap(void) {
  int current_free = kmem.free_pages;
  if(current_free < Th) {
    cprintf("Current Threshold = %d, Swapping %d pages\n", Th, Npg);
    int i = 0;
    while(i < Npg){
      swap_out();
      i = incr(i, 1);
    }
    /////////////////////////
    Th -= (Th * beta) / 100;
    ////////////////////////////
    Npg += (Npg * alpha) / 100;
    if(Npg > LIMIT) Npg = LIMIT;
  }
}


