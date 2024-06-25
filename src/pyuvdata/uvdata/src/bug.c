/************************************************************************/
/*									*/
/* This handles errors and can abort your program.			*/
/*									*/
/*  History:								*/
/*    rjs,mjs ????    Very mixed history. Created, destroyed, rewritten.*/
/*    rjs     26aug93 Call habort_c.					*/
/*    rjs     14jul98 Add a caste operation in errmsg_c, to attempt	*/
/*		      to appease some compilers.			*/
/*    pjt     23sep01 darwin						*/
/*    pjt      4dec01 bypass fatal errors (for alien clients) if req'd  */
/*                    through the new bugrecover_c() routine            */
/*    pjt     17jun02 prototypes for MIR4                               */
/*    pjt/ram  5dec03 using strerror() for unix                         */
/*    pjt      1jan05 bugv_c: finally, a real stdargs version!!!        */
/*                    though cannot be exported to Fortran              */
/************************************************************************/

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "miriad.h"

static char *errmsg_c(int n);

char *Name = NULL;
int reentrant=0;

/*  HPUX cannot handle the (void) thing */

typedef void (*proc)(void);  /* helper definition for function pointers */
static proc bug_cleanup=NULL;

/************************************************************************/
void bugrecover_c(void (*cl)(void))
/** bugrecover_c -- bypass fatal bug calls for alien clients            */
/*& pjt                                                                 */
/*: error-handling                                                      */
/*+                                                                    
    This routine does not have a FORTRAN counterpart, as it is normally 
    only called by C clients who need to set their own error handler if
    for some reason they don't like the MIRIAD one (e.g. C++ or java
    exceptions, or NEMO's error handler 
    Example of usage:

    void my_handler(void) {
        ....
    }


    ..
    bugrecover_c(my_handler);
    ..                                                                  */
/*--                                                                    */
/*----------------------------------------------------------------------*/
{
    bug_cleanup = cl;
}

/************************************************************************/
void buglabel_c(Const char *name)
/** buglabel -- Give the "program name" to be used as a label in messages. */
/*& pjt									*/
/*: error-handling							*/
/*+ FORTRAN call sequence:
	subroutine buglabel(name)

	implicit none
	character name*(*)

  Give the name that is to be used as a label in error messages.

  Input:
    name	The name to be given as a label in error messages.	*/
/*--									*/
/*----------------------------------------------------------------------*/
{
  if(Name != NULL)free(Name);
  Name = malloc(strlen(name)+1);
  strcpy(Name,name);
}
/************************************************************************/
void bug_c(char s,Const char *m)
/** bug -- Issue an error message, given by the caller.			*/
/*& pjt									*/
/*: error-handling							*/
/*+ FORTRAN call sequence:
	subroutine bug(severity,message)

	implicit none
	character severity*1
	character message*(*)

  Output the error message given by the caller, and abort if needed.

  Input:
    severity	Error severity. Can be one of 'i', 'w', 'e' or 'f'
		for "informational", "warning", "error", or "fatal"
    message	The error message text.					*/
/*--									*/
/*----------------------------------------------------------------------*/
{
  char *p;
  int doabort;

  doabort = 0;
  if      (s == 'i' || s == 'I') p = "Informational";
  else if (s == 'w' || s == 'W') p = "Warning";
  else if (s == 'e' || s == 'E') p = "Error";
  else {doabort = 1;		 p = "Fatal Error"; }

  fprintf(stderr,"### %s:  %s\n",p,m);
  if(doabort){
    reentrant = !reentrant;
    if(reentrant)habort_c();
#ifdef vms
# include ssdef
    lib$stop(SS$_ABORT);
#else
/*    fprintf(stderr,"### Program exiting with return code = 1 ###\n"); */
    if (bug_cleanup) {
        (*bug_cleanup)();       /* call it */
        fprintf(stderr,"### bug_cleanup: code should not come here, goodbye\n");
        /* and not it will fall through and exit the bug way */
    }
    exit (1);
#endif
  }
}
/************************************************************************/
void bugv_c(char s,Const char *m, ...)
/** bugv_c -- Issue a dynamic error message, given by the caller.	*/
/*& pjt									*/
/*: error-handling							*/
/*+ C call sequence:
	bugv_c(severity,message,....)

  Output the error message given by the caller, and abort if needed.

  Input:
    severity	Error severity character. 
                Can be one of 'i', 'w', 'e' or 'f'
		for "informational", "warning", "error", or "fatal"
    message	The error message string, can contain %-printf style 
                directives, as used by the following arguments.
     ...         Optional argument, in the printf() style               */
/*--									*/
/*----------------------------------------------------------------------*/
{
  va_list ap;
  char *p;
  int doabort;

  doabort = 0;
  if      (s == 'i' || s == 'I') p = "Informational";
  else if (s == 'w' || s == 'W') p = "Warning";
  else if (s == 'e' || s == 'E') p = "Error";
  else {doabort = 1;		 p = "Fatal Error"; }

  va_start(ap,m);
  fprintf(stderr,"### %s: ",p);
  vfprintf(stderr,m,ap);
  fprintf(stderr,"\n");     /* should *we* really supply the newline ? */
  fflush(stderr);
  va_end(ap);

  if(doabort){
    reentrant = !reentrant;
    if(reentrant)habort_c();
    if (bug_cleanup) {
        (*bug_cleanup)();       /* call it */
        fprintf(stderr,"### bug_cleanup: code should not come here, goodbye\n");
        /* and not it will fall through and exit the bug way */
    }
    exit (1);
  }
}

/************************************************************************/
void bugno_c(char s,int n)
/** bugno -- Issue an error message, given a system error number.	*/
/*& pjt									*/
/*: error-handling							*/
/*+ FORTRAN call sequence:
	subroutine bugno(severity,errno)

	implicit none
	character severity*1
	integer errno

  Output the error message associated with a particular error number.

  Input:
    severity	Error severity. Can be one of 'i', 'w', 'e' or 'f'
		for "informational", "warning", "error", or "fatal"
    errno	host error number.					*/
/*--									*/
/*----------------------------------------------------------------------*/
{
  if (n == -1)bug_c(s,"End of file detected");
  else bug_c(s,errmsg_c(n));
}
/************************************************************************/
static char *errmsg_c(int n)
/*
  Return the error message associated with some error number.
------------------------------------------------------------------------*/
{
#ifdef vms
#include <descrip.h>
  $DESCRIPTOR(string_descriptor,string);
  static char string[128];
  short int len0;
  int one;

  one = 1;
  lib$sys_getmsg(&n,&len0,&string_descriptor,&one);
  string[len0] = 0;
  return(string);
#else
  return strerror(n);
#endif
}
