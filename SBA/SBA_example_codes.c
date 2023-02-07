#include <stdio.h>
#include <string.h>

#define MAX_LENGTH 100
#define ALPHABET_SIZE 26

int count_frequency(char text[MAX_LENGTH]) {
  int frequency[ALPHABET_SIZE] = {0};
  int length = strlen(text);
  int i;

  for (i = 0; i < length; i++) {
    char c = text[i];
    if (c >= 'A' && c <= 'Z') {
      frequency[c - 'A']++;
    }
  }

  return frequency;
}


char shift_cipher(char c, int shift) {
  if (c >= 'A' && c <= 'Z') {
    return (c - 'A' + shift) % ALPHABET_SIZE + 'A';
  } else {
    return c;
  }
}

void encrypt(char text[MAX_LENGTH], int shift) {
  int length = strlen(text);
  int i;
  for (i = 0; i < length; i++) {
    text[i] = shift_cipher(text[i], shift);
  }
}

void decrypt(char text[MAX_LENGTH], int shift) {
  int length = strlen(text);
  int i;
  for (i = 0; i < length; i++) {
    text[i] = shift_cipher(text[i], ALPHABET_SIZE - shift);
  }
}

int main() {
  char text[MAX_LENGTH];
  int shift;
  printf("Enter a text: ");
  scanf("%s", text);
  printf("Enter a shift: ");
  scanf("%d", &shift);
  encrypt(text, shift);
  printf("Encrypted text: %s\n", text);
  decrypt(text, shift);
  printf("Decrypted text: %s\n", text);
  return 0;
}

int main() {
  char text[MAX_LENGTH];
  printf("Enter a text: ");
  scanf("%s", text);
  int *frequency = count_frequency(text);
  int i;
  for (i = 0; i < ALPHABET_SIZE; i++) {
    printf("%c: %d\n", 'A' + i, frequency[i]);
  }
  return 0;
}