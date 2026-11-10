#include "/home/pablo_kevin/Arduino/Projects/esp32_eezybot_linear/esp32_eezybot_linear.ino"
void L(const char*s){char b[96];strcpy(b,s);printf("> %s\n",s);handleLine(b);}
void run(){for(int i=0;i<600;i++){fake_us+=20000;motionTick();}}
int main(){ setup(); L("PUSH1"); run(); L("PUSH2"); run(); L("push3"); run(); printPos(); L("PUSH 3"); run(); L("PUSH4");}
