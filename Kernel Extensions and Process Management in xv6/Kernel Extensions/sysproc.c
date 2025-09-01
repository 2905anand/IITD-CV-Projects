#include "types.h"
#include "x86.h"
#include "defs.h"
#include "date.h"
#include "param.h"
#include "memlayout.h"
#include "mmu.h"
#include "proc.h"
#include "stdint.h"

int
sys_fork(void)
{
  return fork();
}

int
sys_exit(void)
{
  exit();
  return 0;  // not reached
}

int
sys_wait(void)
{
  return wait();
}

int
sys_kill(void)
{
  int pid;

  if(argint(0, &pid) < 0)
    return -1;
  return kill(pid);
}

int
sys_getpid(void)
{
  return myproc()->pid;
}

int
sys_sbrk(void)
{
  int addr;
  int n;

  if(argint(0, &n) < 0)
    return -1;
  addr = myproc()->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

int
sys_sleep(void)
{
  int n;
  uint ticks0;

  if(argint(0, &n) < 0)
    return -1;
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(myproc()->killed){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

// return how many clock tick interrupts have occurred
// since start.
int
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

// sysproc.c
int sys_gethistory(void) {
  struct history_entry *user_buf;
  int n;

  // Fetch user buffer and size
  if (argptr(0, (void*)&user_buf, sizeof(struct history_entry) * MAX_HISTORY_ENTRIES) < 0 ||
      argint(1, &n) < 0) {
    return -1;
  }

  // Copy history to user space
  acquire(&history_lock);
  for (int i = 0; i < history_index && i < n; i++) {
    if (copyout(myproc()->pgdir, (uintptr_t)&user_buf[i], (char*)&proc_history[i], sizeof(struct history_entry)) < 0) {
      release(&history_lock);
      return -1;
    }
  }
  release(&history_lock);
  return history_index;
}


int sys_block(void) {
  struct proc *curproc = myproc();
	int syscall_id;
	
	if(argint(0, &syscall_id) < 0){
		return -1;
	}
	//cprintf("%d \n", syscall_id);
  // Block critical syscalls (fork=1, exit=2)
  if (syscall_id == 1 || syscall_id == 2)
    return -1;

  if (syscall_id < 0 || syscall_id >= MAX_SYSCALLS)
    return -1;

  curproc->system_calls_block_status[syscall_id] = 1;
  return 0;
}

int sys_unblock(void) {
  struct proc *curproc = myproc();
  int syscall_id;
	
	if(argint(0, &syscall_id) < 0){
		return -1;
	}

  if (syscall_id < 0 || syscall_id >= MAX_SYSCALLS)
    return -1;

  curproc->system_calls_block_status[syscall_id] = 0;
  return 0;
}
