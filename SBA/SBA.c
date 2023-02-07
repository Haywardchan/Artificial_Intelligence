#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
int enter;
char go;
int randomnumber;

int hubscrean(){
    printf("Welcome!\nThis is a Shift Ciphers Decrypter\nPress [Enter] to start ");
    scanf("%c",&go);
    return 0;
}

int instrections(){
    printf("\nInstrections\n");
    printf("1. You can ONLY enter UPPER CASE LETTER, space characters and punctuation marks.\n");
    printf("2. Space characters and puncturation marks remain unchanged during the encryption.\n");
    printf("Choose whitch function you want to use: ([1] Encrypt/ [2] Decrypt)\nEnter your choice here [1/2]: ");
    scanf("%d",&enter);
    while(enter!=1&&enter!=2){
        printf("Plz reenter your choice again [1 Encrypt/2 Decrypt]: ");
        scanf("%d",&enter);
    }
    return 0;
}

int encrypt(){
    char input[90000];
    char store[90000];
    char temp;
    int shift;
    //printf("%s",store);
    re:;
    printf("---------------------------------------\nEnter your passage below\n---------------------------------------\n");
    scanf("%c",&temp); // temp statement to clear buffer
    scanf("%[^\n]",input);
    for(int i=0;i<90000;i++){
        store[i]=input[i];
    }
    int c=0;
    while (store[c]!='\0'){
      if(store[c]>='a' && store[c]<='z'){
        printf("\n---------------------------------------\n");
        printf("Small case leter cant be shifted, you must enter UPPER CASE letter to be able to do so.\n");
        printf("Press [Enter] to contiue, or [1] to reenter.");
        char enter2=0;
        temp=0;
        scanf("%c",&temp); // temp statement to clear buffer
        scanf("%c",&enter2);
        if(enter2=='1'){
          goto re;
        }
        goto next2;
      }
      c++;
    }
  next2:;
    printf("\n---------------------------------------\nEnter Shift (number) or [0] to Shift random: ");
    scanf("%d",&shift);
    printf("\n---------------------------------------");
  // Count the letter frequencies of the input
    c=0;
    int count[26]={0};
    int x=0;
    int max=0;
    char letter=0;
    while (store[c]!='\0'){
        if (store[c]>='A' && store[c]<='Z') {
            x=store[c]-'A';
            count[x]++;
        }
        c++;
    }
    for (c = 0; c < 26; c++){
        if(count[c]>max){
            max=count[c];
            letter='A'+c;
        }
    }
    printf("\nThe letter %c occurs most, occured %d times in the message.",letter,max);
    printf("\n");
    for (c = 0; c < 26; c++){
        printf("%c occurs %d times in the string.\n", c + 'A', count[c]);
    }
    // Encrypt
    int printshift=0;
    if(shift==0){
      shift=randomnumber;
    }
    printshift=shift;
    while(shift<0){
      shift=26+shift;
    }
    for(int i=0;(i<90000 && input[i]!='\0');i++){
        if(input[i]>=65 && input[i]<=90){
        input[i]=(input[i]+shift-'A')%26+'A';
        }
    }
    printf("\nOUTPUT (Shift %d)\n---------------------------------------\n%s\n",printshift,input);
    
    return 0;
}

int decrypt(){
    char input[90000];
    char store[90000];
    char store2[90000];
    char temp;
    int shift=1;
    printf("---------------------------------------\nEnter your passage below\n---------------------------------------\n");
    scanf("%c",&temp); // temp statement to clear buffer
    scanf("%[^\n]",input); // read the input
    for(int i=0;i<90000;i++){ // copy the string for Counting the letter frequencies
    store[i]=input[i];
    store2[i]=input[i];
    }
    
    // Count the letter frequencies of the input
    int c=0;
    int count[26]={0};
    int x=0;
    int max=0;
    char letter=0;
    while (store[c]!='\0'){
        if (store[c]>='A' && store[c]<='Z') {
            x=store[c]-'A';
            count[x]++;
        }
        c++;
    }
    for (c = 0; c < 26; c++){
        if(count[c]>max){
            max=count[c];
            letter='A'+c;
        }
    }
    printf("\n---------------------------------------\n\nLetter frequencies of the cipher text.");
    printf("\n---------------------------------------\n");
    printf("The letter %c occurs most, occured %d times in the cipher text.\n",letter,max);
    for (c = 0; c < 26; c++){
        printf("%c occurs %d times in the string.\n", c + 'A', count[c]);
    }
    printf("---------------------------------------\n");
    int countshift=0;
    countshift=letter-'E';
    printf("\nThe guessed message is\n---------------------------------------\n");
    for(int i=0;(i<90000&& input[i] != '\0');i++){
        if(input[i]>=65 && input[i]<=90){
        store2[i]=(input[i]+26-countshift-'A')%26+'A';
        }
    }
    printf("Shift %d\n\n",-countshift);
    printf("%s\n",store2);
    back:;
    printf("\n---------------------------------------\n");
    printf("Press [1] to show all result\n");
    printf("Press [2] to show other top 5 guesed result\n");
    printf("Press [Enter] to Exit or enter [3] to return to main screan\n");
    scanf("%c",&temp); // temp statement to clear buffer
    scanf("%c",&go);
    printf("---------------------------------------\n");
    if(go!='1'&&go!='2'){
        return 0;
    }
    if(go=='1'){
        shift=1;
        printf("---------------------------------------\n\nOther shifted message\n---------------------------------------\n");
        for(int j=1;j<26;j++){ // repeat the shifting progress 26 times
            for(int i=0;(i<90000&& input[i] != '\0');i++){ // keep shifting each letter until there is no letter available to shift
                if(input[i]>=65 && input[i]<=90){   // filter out non uppercase letter
                    store2[i]=(input[i]+j-'A')%26+'A'; // shift the uper case letter by j
                }
            }
        printf("\nShift %d\n",shift); // print out the number of shift of the answer
        printf("%s\n",store2); // print out all possibe answer
        shift++;
    }
    }
    if(go=='2'){
        countshift=0;
        int No=0;
        char c[5]={'E','A','R','I','O'};
        printf("\nThe Top 5 guessed message is\n---------------------------------------\n");
        for(No=0;No<5;No++){
            countshift=0;
            printf("No.%d   ",No+1);
            countshift=letter-c[No];
            for(int i=0;(i<90000&& input[i] != '\0');i++){
                if(input[i]>=65 && input[i]<=90){
                    store2[i]=(input[i]+26-countshift-'A')%26+'A';
                }
            }
            printf("Shift %d\n\n",-countshift);
            printf("%s\n",store2);
            printf("\n---------------------------------------\n");
        }
    }
    goto back;
    return 0;
}

int endscrean(){
    char temp;
    printf("\n---------------------------------------\nPress [Enter] to Exit or enter [1] to return to main screan.\n");
    scanf("%c",&temp); // temp statement to clear buffer
    scanf("%c",&go);
    return 0;
}

int main(){
    hubscrean();
    retry:;
    // Generates random numbers in range [lower, upper].
    int lower = 1, upper = 25, count = 1;
    srand(time(0));
    int i;
    for (i = 0; i < count; i++) {
      randomnumber = (rand() %
      (upper - lower + 1)) + lower;
      //printf("%d ", randomnumber);
    }
    instrections();
    if(enter==1){
        encrypt();
    }
    if(enter==2){
        decrypt();
        if(go!='1'&&go!='2'&&go!='3'){
            printf("Thx for using");
            return 0;
        }
        if(go=='3'){
            printf("\n---------------------------------------\n");
        goto retry;
    }
    }
    endscrean();
    if(go=='1'){
        printf("\n---------------------------------------\n");
        goto retry;
    }
    printf("Thx for using");
    return 0;
}