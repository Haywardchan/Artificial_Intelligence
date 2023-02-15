#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#define _RED "\x1b[31m"
#define _GREEN "\x1b[32m"
#define _BLUE "\x1b[;34m"
#define _MAGENTA "\x1b[35m"
#define _WHITE "\x1b[37m"
#define _RESET "\x1B[0m"
#define _BOLD "\x1B[1m"
#define _V1 "\x1b[4;3;34m"
#define _V2 "\x1b[3;32m"
#define _V3 "\x1B[1m\x1b[3;31m"

int enter;
int randomnumber;
int shift;
char input[90000];
char store[90000];
char letter=0;
char temp;
char go;

int generate_randomnumber() {
  srand(time(0));
  int shift = rand();
  return shift % 26;
}

int countletterfrequencies(){
    int x=0;
    int max=0;
    int c=0;
    int count[26]={0};
    letter=0;
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
    printf("\n----------------------------------------------------------\n\n%sLetter frequencies of the message.%s",_BOLD,_RESET);
    printf("\n----------------------------------------------------------");
    printf("\nThe letter %c occurs most, occured %d times in the message.\n",letter,max);
    printf("\n");
    for (c = 0; c < 26; c++){
        printf("%c occurs %d times in the string.\n", c + 'A', count[c]);
    }
    return 0;
}

int hubscreen(){
    printf("%s%sWELCOME!\n%s%sThis is a Shift Ciphers Decrypter\nPress %s[Enter]%s to start ",_V1,_BOLD,_RESET,_WHITE,_BOLD,_RESET);
    scanf("%c",&go);
    return 0;
}

int instrections(){
    printf("\nInstrections\n");
    printf("1. You can ONLY enter %sUPPER CASE LETTER%s, space characters and punctuation marks.\n",_BOLD,_RESET);
    printf("2. Space characters and puncturation marks remain unchanged during the encryption.\n");
    printf("Choose witch function you want to use: %s([1] Encrypt / [2] Decrypt)%s\nEnter your choice here %s[1/2]%s: ",_BOLD,_RESET,_BOLD,_RESET);
    scanf("%d",&enter);
    while(enter!=1&&enter!=2){
        printf("%sPlz reenter your choice again %s[1 Encrypt/2 Decrypt]:%s ",_RED,_V3,_RESET);
        scanf("%d",&enter);
    }
    return 0;
}

int encrypt(){
    bool flag;
    re:;
    printf("----------------------------------------------------------\n%sEnter your passage below%s\n----------------------------------------------------------\n",_BOLD,_RESET);
    scanf("%c",&temp); // temp statement to clear buffer
    scanf("%[^\n]",input);
    for(int i=0;i<90000;i++){
        store[i]=input[i];
    }
    int c=0;
    while (store[c]!='\0'){
      if(store[c]>='a' && store[c]<='z'){
        printf("\n----------------------------------------------------------\n");
        printf("%sSmall case leter cant be shifted, you %sMUST%s%s enter %sUPPER CASE%s%s letter to be able to do so.%s\n",_RED,_V3,_RESET,_RED,_V3,_RESET,_RED,_RESET);
        printf("Press %s[Enter]%s to contiue, or %s[1]%s to reenter.",_BOLD,_RESET,_BOLD,_RESET);
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
    printf("\n----------------------------------------------------------\n%sEnter Shift (number) or [0] to Shift random:%s ",_BOLD,_RESET);
    scanf("%d",&shift);
    countletterfrequencies();
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
    printf("\n%sOUTPUT (Shift %d)%s\n----------------------------------------------------------\n%s\n",_BOLD,printshift,_RESET,input);
    return 0;
}

int decrypt(){
    int shift=1;
    printf("----------------------------------------------------------\n%sEnter your passage below%s\n----------------------------------------------------------\n",_BOLD,_RESET);
    scanf("%c",&temp); // temp statement to clear buffer
    scanf("%[^\n]",input); // read the input
    for(int i=0;i<90000;i++){ // copy the string for Counting the letter frequencies
    store[i]=input[i];
    }
    countletterfrequencies(); //count letter frequencies
    printf("----------------------------------------------------------\n");
    for(int i=0;i<90000;i++){ // copy the string for shifting
    store[i]=input[i];
    }
    int countshift=0;
    countshift=letter-'E'; // set countshift to be the most frequenct letter - 'E'
    printf("\n%sThe guessed message is%s\n----------------------------------------------------------\n",_BOLD,_RESET);
    for(int i=0;(i<90000&& input[i] != '\0');i++){ // keep shifting the letter until no letter can be read
        if(input[i]>=65 && input[i]<=90){    // filter out non uppercase letter
        store[i]=(input[i]+26-countshift-'A')%26+'A'; // shift the uper case letter by countshift
        }
    }
    printf("Shift %d\n\n",-countshift);
    printf("%s\n",store);
    back:; //return to the decictions
    printf("\n----------------------------------------------------------\n");
    printf("Press %s[1]%s to show all result\n",_BOLD,_RESET);
    printf("Press %s[2]%s to show other top 5 guesed result\n",_BOLD,_RESET);
    printf("Press %s[Enter]%s to Exit or enter %s[3]%s to return to main screan\n",_BOLD,_RESET,_BOLD,_RESET);
    scanf("%c",&temp); // temp statement to clear buffer
    scanf("%c",&go); // read the decictions
    printf("----------------------------------------------------------\n");
    if(go!='1'&&go!='2'){ // read input choice
        return 0; // exit decrypt and goto endscreen
    }
    if(go=='1'){ // read input choice
        shift=25;
        printf("----------------------------------------------------------\n\n%sOther shifted message%s\n----------------------------------------------------------\n",_BOLD,_RESET);
        for(int j=26;j!=1;j--){ // repeat the shifting progress 26 times
            for(int i=0;(i<90000&& input[i] != '\0');i++){ // keep shifting the letter until no letter can be read
                if(input[i]>=65 && input[i]<=90){   // filter out non uppercase letter
                    store[i]=(input[i]+j-1-'A')%26+'A'; // shift the uper case letter by j
                }
            }
        printf("\nShift %d\n",shift); // print out the number of shift of the result
        printf("%s\n",store); // print out all possibe result
        shift--;
    }
    }
    if(go=='2'){ //read input choice
        countshift=0; //reset countshift
        int No=0;
        char c[5]={'O','I','R','A','E'}; //guses the top 5 message by the top 5 most frequent letter in a long english paragraph
        printf("\n%sThe Top 5 guessed message is%s\n----------------------------------------------------------\n",_BOLD,_RESET);
        for(No=0;No<5;No++){ // repeat the process 5 times
            countshift=0; // reset count shift
            printf("No.%d   ",5-No);
            countshift=letter-c[No]; // set countshift to the most frequent letter - 'E'/'A'/'R'/'I'/'O'
            for(int i=0;(i<90000&& input[i] != '\0');i++){ // keep shifting the letter until no letter can be read
                if(input[i]>=65 && input[i]<=90){ // filter out non uppercase letter
                    store[i]=(input[i]+26-countshift-'A')%26+'A'; // shift letter by countshift
                }
            }
            printf("Shift %d\n\n",-countshift); //print out the number of shift of the result
            printf("%s\n",store); // print out the result
            printf("\n----------------------------------------------------------\n");
        }
    }
    goto back; //return to the decictions
    return 0;
}

int endscreen(){
    char temp;
    printf("\n----------------------------------------------------------\nPress %s[Enter]%s to Exit or enter %s[1]%s to return to main screan.\n",_BOLD,_RESET,_BOLD,_RESET);
    scanf("%c",&temp); // temp statement to clear buffer
    scanf("%c",&go);
    return 0;
}

int main(){
    hubscreen();
    retry:;
    randomnumber=generate_randomnumber();
    instrections();
    if(enter==1){
        encrypt();
    }
    if(enter==2){
        decrypt();
        if(go!='1'&&go!='2'&&go!='3'){
            printf("%s%sThx for using%s",_BOLD,_V2,_RESET);
            return 0;
        }
        if(go=='3'){
            printf("\n----------------------------------------------------------\n");
            goto retry;
        }
    }
    endscreen();
    if(go=='1'){
        printf("\n----------------------------------------------------------\n");
        goto retry;
    }
    printf("%s%sThx for using%s",_BOLD,_V2,_RESET);
    return 0;
}