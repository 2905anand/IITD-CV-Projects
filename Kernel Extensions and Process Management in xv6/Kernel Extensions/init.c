// init: The initial user-level program

#include "types.h"
#include "stat.h"
#include "user.h"
#include "fcntl.h"

char *argv[] = { "sh", 0 };

void trim(char *string) {
  char *s = string;
  while (*s) {
    if (*s == '\n' || *s == '\r') {
      *s = '\0';
      return;
    }
    s++;
  }
}

int
main(void)
{
  int pid, wpid;
  char username[32], password[32];
  int attempts = 3;
  int verified = 0;

  if(open("console", O_RDWR) < 0){
    mknod("console", 1, 1);
    open("console", O_RDWR);
  }
  dup(0);  // stdout
  dup(0);  // stderr
  
  while(attempts>0 && verified==0){

    printf(1, "Enter username: ");
    gets(username, sizeof(username));
    trim(username);
    if(strcmp(username, USERNAME) != 0){
    	printf(1, "Invalid Username. Try again.\n");
    	attempts--;
    	continue;
    }
  
  	printf(1, "Enter password: ");
  	gets(password, sizeof(password));
  	trim(password);
		if (strcmp(password, PASSWORD) != 0) {
      printf(1, "Invalid Password. Try again\n");
      attempts--;
      continue;
    }
    
    verified = 1;
  }

  // If attempts are exhausted, disable login
  if (verified == 0) {
    printf(1, "3 failed attempts. Login disabled.\n");
    exit();  // Exit the program if failed too many times
  }
  
  //printf(1, "Login Successful\n");
  
  for(;;){
    printf(1, "init: starting sh\n");
    pid = fork();
    if(pid < 0){
      printf(1, "init: fork failed\n");
      exit();
    }
    if(pid == 0){
      exec("sh", argv);
      printf(1, "init: exec sh failed\n");
      exit();
    }
    while((wpid=wait()) >= 0 && wpid != pid)
      printf(1, "zombie!\n");
  }
}
