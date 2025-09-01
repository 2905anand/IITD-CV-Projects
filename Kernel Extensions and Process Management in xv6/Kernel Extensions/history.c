// history.c
#include "types.h"
#include "user.h"
#include "stat.h"

#define MAX_ENTRIES 64

int main() {
  struct history_entry entries[MAX_ENTRIES];
  int count = gethistory(entries, MAX_ENTRIES);

  if (count < 0) {
    printf(2, "Error retrieving history\n");
    exit();
  }

  // Print entries sorted by creation time (ascending)
  for (int i = 0; i < count; i++) {
    printf(1, "%d %s %d\n", entries[i].pid, entries[i].name, entries[i].mem_util);
  }

  exit();
}
