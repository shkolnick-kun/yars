#include <stdio.h>
#include <stdlib.h>

#include <yars.h>

int main()
{
    printf("%f \n", yasr_weight(&yars_defaults, 0));
    return 0;
}
