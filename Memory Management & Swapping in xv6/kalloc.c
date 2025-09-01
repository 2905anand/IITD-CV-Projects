// Physical memory allocator, intended to allocate
// memory for user processes, kernel stacks, page table pages,
// and pipe buffers. Allocates 4096-byte pages.

#include "types.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "mmu.h"
#include "spinlock.h"
#include "pageswap.h"

void freerange(void *vstart, void *vend);
extern char end[]; // first address after kernel loaded from ELF file
                   // defined by the kernel linker script in kernel.ld

struct run {
  struct run *next;
};



struct {
  struct spinlock lock;
  int use_lock;
  struct run *freelist;
  int free_pages;
} kmem;


void
initls(void)
{
  return;
}

// Initialization happens in two phases.
// 1. main() calls kinit1() while still using entrypgdir to place just
// the pages mapped by entrypgdir on free list.
// 2. main() calls kinit2() with the rest of the physical pages
// after installing a full page table that maps them on all cores.
void
kinit1(void *vstart, void *vend)
{
  initlock(&kmem.lock, "kmem");
  kmem.use_lock = 0;
  kmem.free_pages = 0;
  freerange(vstart, vend);
  cprintf("free_pages at the time of kinit1 : %d",kmem.free_pages);

}

void
kinit2(void *vstart, void *vend)
{
  freerange(vstart, vend);
  kmem.use_lock = 1;

  ///////////////////////////////////////////////////////
  struct run *r = kmem.freelist;
  kmem.free_pages = 0;
  ////////////////////////////////////////////////////
  while(r) {
    kmem.free_pages = kmem.free_pages + 1;
    initls();
    r = r->next;
  }
  cprintf("free_pages at kinit2 : %d",kmem.free_pages);
  /////////////////////////////////////////////////////
}

void
freerange(void *vstart, void *vend)
{
  char *p;
  p = (char*)PGROUNDUP((uint)vstart);
  for(; p + PGSIZE <= (char*)vend; p += PGSIZE)
    kfree(p);
}
//PAGEBREAK: 21
// Free the page of physical memory pointed at by v,
// which normally should have been returned by a
// call to kalloc().  (The exception is when
// initializing the allocator; see kinit above.)
void
kfree(char *v)
{
  struct run *r;

  if((uint)v % PGSIZE || v < end || V2P(v) >= PHYSTOP)
    panic("kfree");

  // Fill with junk to catch dangling refs.
  memset(v, 1, PGSIZE);

  if(kmem.use_lock)
    acquire(&kmem.lock);
  r = (struct run*)v;
  r->next = kmem.freelist;
  kmem.freelist = r;
  if(!kmem.free_pages){
  }else if(kmem.free_pages){
    kmem.free_pages++;
  }
  if(kmem.use_lock)
    release(&kmem.lock);

}






void unlock_(){
  if(kmem.use_lock)
    release(&kmem.lock);
}

void lock_(){
  if(kmem.use_lock)
    acquire(&kmem.lock);
}


char* allocation(struct run * r){
      /////////////////////////////////////////////
      kmem.freelist = r->next;
      // if(kmem.use_lock)
      //   release(&kmem.lock);
      unlock_();
      ///////////////////////////////////////
      int size = 1;
      int pg_size  = PGSIZE;
      memset((char*)r, 5*size, pg_size);
      kmem.free_pages = kmem.free_pages - 1;
      if(kmem.free_pages<=0){
      }else if(kmem.free_pages>0){
        adaptive_swap();
      }else{
      }
      return (char*)r;
      /////////////////////////////////////
}


char*
kalloc(void)
{
  struct run *r;
  lock_();
  r = kmem.freelist;
  if(r) {
    return allocation(r);
  }
  unlock_();
  swap_out();
  lock_();
  r = kmem.freelist;
  if(r) {
    return allocation(r);
  }

  unlock_();
  panic("kalloc: out of memory and swap failed");
}


