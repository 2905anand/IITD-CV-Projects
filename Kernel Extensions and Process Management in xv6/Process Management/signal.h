// #include "memlayout.h"   // for KERNBASE and PGSIZE
// #define SIGRETURN_ADDR (KERNBASE - PGSIZE)


// // In user.h, add these definitions (if not already present):
// #define SIGINT    2  // Ctrl+C signal (commonly 2 in Unix)
// #define SIGBG     3  // Ctrl+B signal (for background suspension)
// #define SIGFG     4  // Ctrl+F signal (for foreground/resume)
// #define SIGCUSTOM 5  // Ctrl+G signal (for custom handler)


// extern void sendsig(int sig);
// extern volatile int ctrlc_requested;
// extern volatile int print_ctrlc_flag;