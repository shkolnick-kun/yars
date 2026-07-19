#include <stdio.h>
#include <stdlib.h>

#include <yars.h>

int main()
{
    printf("%f \n", yars_f32_weight(&yars_f32_defaults, 0));
    return 0;
}
