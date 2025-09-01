#include "x86.h"


#include "memlayout.h"   // for KERNBASE and PGSIZE
#define SIGRETURN_ADDR (KERNBASE - PGSIZE)


// In user.h, add these definitions (if not already present):
#define SIGINT    2  // Ctrl+C signal (commonly 2 in Unix)
#define SIGBG     3  // Ctrl+B signal (for background suspension)
#define SIGFG     4  // Ctrl+F signal (for foreground/resume)
#define SIGCUSTOM 5  // Ctrl+G signal (for custom handler)


extern void sendsig(int sig);
extern volatile int ctrlc_requested;
extern volatile int print_ctrlc_flag;
extern volatile int ctrlg_requested;

// Per-CPU state
struct cpu {
  uchar apicid;                // Local APIC ID
  struct context *scheduler;   // swtch() here to enter scheduler
  struct taskstate ts;         // Used by x86 to find stack for interrupt
  struct segdesc gdt[NSEGS];   // x86 global descriptor table
  volatile uint started;       // Has the CPU started?
  int ncli;                    // Depth of pushcli nesting.
  int intena;                  // Were interrupts enabled before pushcli?
  struct proc *proc;           // The process running on this cpu or null
};

extern struct cpu cpus[NCPU];
extern int ncpu;

//PAGEBREAK: 17
// Saved registers for kernel context switches.
// Don't need to save all the segment registers (%cs, etc),
// because they are constant across kernel contexts.
// Don't need to save %eax, %ecx, %edx, because the
// x86 convention is that the caller has saved them.
// Contexts are stored at the bottom of the stack they
// describe; the stack pointer is the address of the context.
// The layout of the context matches the layout of the stack in swtch.S
// at the "Switch stacks" comment. Switch doesn't save eip explicitly,
// but it is on the stack and allocproc() manipulates it.
struct context {
  uint edi;
  uint esi;
  uint ebx;
  uint ebp;
  uint eip;
};

enum procstate { UNUSED, EMBRYO, SLEEPING, RUNNABLE, RUNNING, ZOMBIE, CREATED };

typedef void (*sighandler_t)(void);

// Per-process state
struct proc {
  uint sz;                     // Size of process memory (bytes)
  pde_t* pgdir;                // Page table
  char *kstack;                // Bottom of kernel stack for this process
  enum procstate state;        // Process state
  int pid;                     // Process ID
  struct proc *parent;         // Parent process
  struct trapframe *tf;        // Trap frame for current syscall
  struct context *context;     // swtch() here to run process
  void *chan;                  // If non-zero, sleeping on chan
  int killed;                  // If non-zero, have been killed
  struct file *ofile[NOFILE];  // Open files
  struct inode *cwd;           // Current directory
  char name[16];               // Process name (debugging)
  
  int sigsuspended; //Changes
  int start_later_flag;
  int exec_time;
  int start_ticks;
  int ticks_run;
  int creation_time;
  int run_time;
  int wait_time;
  // if received sigint signal-------------------------
  int in_handler; ///
  int sigint_killed;
  sighandler_t custom_handler_function;   
  int pending_custom_handler;
  void (*signal_handler)(void);
  //-----------------------------------------  
  int end_time;
  int CS;
  int priority;            // Initial priority (set to INIT_PRIORITY)
  int dynamic_priority;    // Computed dynamic priority (πi(t))
};

// Process memory is laid out contiguously, low addresses first:
//   text
//   original data and bss
//   fixed-size stack
//   expandable heap
