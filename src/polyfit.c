#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_statistics.h>

#define VERSION "2021-03-16"

#define DEBUG 0

/* Set the defaults; you can override these from the command line.
int    POLYORDER = 2;
int    ITERS     = 10000;
double STEP_SIZE = 0.01;
int    NVERT = 200;
int    KNOTS = 4;
int    COLIDX[3] = {-1, -1, -1};
int    FIND_KNOTS = 0;
int    FIND_STEP = 0;

int    CHAIN_LENGTH = 8;

int    APPLY_PSHIFT = 0;
*/

int    CHAIN_LENGTH;
int    KNOTS;
int    POLYORDER;
int    NVERT;
int    APPLY_PSHIFT;

int    CSV_INPUT = 0;
int    CSV_OUTPUT = 0;
int    EBAI_OUTPUT = 0;
int    SUMMARY_OUTPUT = 0;

char buffer[256];
char *messages = "";

/*
double LLIMIT = -1e20;
double ULIMIT = 1e20;
 */
/* Maximum number of observed data points: */
//#define CHUNK 10

typedef struct triplet {
    double x;
    double y;
    double z;
} triplet;

char * combine( const char *s1, const char *s2 )
{
    char *result = malloc( strlen( s1 ) + strlen( s2 ) + 1 );

    if ( result )
    {
        char *p = result;

        while ( *s1 ) *p++ = *s1++;

        while ( ( *p++ = *s2++ ) );
    }

    return result;
}

int sort_by_phase (const void *a, const void *b)
{
    const triplet *da = (const triplet *) a;
    const triplet *db = (const triplet *) b;

    return (da->x > db->x) - (da->x < db->x);
}

int sort_by_value (const void *a, const void *b)
{
    const double *da = (const double *) a;
    const double *db = (const double *) b;

    return (*da > *db) - (*da < *db);
}

int find_knots (triplet *data, int nobs, double **knots)
{
    int i, j;
    double average = 0.0;
    double std = 0.0;
    int chain_too_short;
    int chain_wrapped = 0;

    int chains = 0;
    struct {
        int len;
        int start;
        int end;
    } chain[2];

    for (i = 0; i < nobs; i++)
        average += data[i].y;
    average /= (double) nobs;

    for (i = 0; i < nobs; i++)
        std += pow(data[i].y - average, 2.0);
    std /= (double) (nobs-1);
    std = sqrt(std);

    /* Find eclipses below this limit */
    average = average - std/2.0;

    if (!SUMMARY_OUTPUT && !EBAI_OUTPUT) {
        sprintf (buffer,"# searching for knots automatically:\n");
        messages = combine(messages ,buffer);
        sprintf (buffer,"# * average-std value of the flux: %lf\n", average);
        messages = combine(messages ,buffer);
    }

    /* To allow wrapping of the phase interval, skip the first chain: */
    i = 0;
    while (data[i].y < average) {
        if (DEBUG)
            sprintf (buffer,"# delaying point  %8.4lf (%3d): %lf < %lf\n", data[i].x, i, data[i].y, average);
        i++;
    }

    for ( ; i < nobs; i++) {
        if (data[i].y > average) {
            if (DEBUG)
                sprintf (buffer,"# skipping point  %8.4lf (%3d): %lf > %lf\n", data[i].x, i, data[i].y, average);
            continue;
        }

        /* Check if the chain is at least CHAIN_LENGTH long: */
        chain_too_short = 0;
        if (DEBUG)
            sprintf (buffer,"# chain starts at %8.4lf (%3d): %lf < %lf\n", data[i].x, i, data[i].y, average);
        for (j = 1; j < CHAIN_LENGTH; j++) {
            if (i+j == nobs) {
                i = -j;
                chain_wrapped = 1;
            }
            if (data[i+j].y > average) {
                if (DEBUG)
                    sprintf (buffer,"# chain broken at %8.4lf (%3d): %lf > %lf\n", data[i+j].x, i+j, data[i+j].y, average);
                i += j;
                chain_too_short = 1;
                break;
            }
            else
                if (DEBUG)
                    sprintf (buffer,"# chain cont'd at %8.4lf (%3d): %lf < %lf\n", data[i+j].x, i+j, data[i+j].y, average);
        }

        if (chain_wrapped && chain_too_short) break;
        if (chain_too_short) continue;

        while (data[i+j].y < average) {
            if (i+j == nobs) {
                i = -j;
                chain_wrapped = 1;
            }
            if (DEBUG)
                sprintf (buffer,"# chain cont'd at %8.4lf (%3d): %lf < %lf\n", data[i+j].x, i+j, data[i+j].y, average);
            j++;
        }

        if (chains < 2) {
            chains++;
            chain[chains-1].len = j-1;
            chain[chains-1].start = chain_wrapped*nobs + i;
            chain[chains-1].end = i+j-1;
        }
        else {
            chains++; /* Just to count all chains */
            if (j-1 > chain[0].len) {
                if (chain[0].len > chain[1].len) {
                    chain[1].len = chain[0].len;
                    chain[1].start = chain[0].start;
                    chain[1].end = chain[0].end;
                }
                chain[0].len = j-1;
                chain[0].start = chain_wrapped*nobs + i;
                chain[0].end = i+j-1;
            }
            else if (j-1 > chain[1].len) {
                chain[1].len = j-1;
                chain[1].start = chain_wrapped*nobs + i;
                chain[1].end = i+j-1;
            }
            /* else drop through without recording it because it is shorter. */
        }

        if (!SUMMARY_OUTPUT && !EBAI_OUTPUT){
            sprintf (buffer,"# * found a chain from %lf (index %d/%d) to %lf (index %d/%d)\n", data[chain_wrapped*nobs+i].x, chain_wrapped*nobs+i, nobs-1, data[i+j-1].x, i+j-1, nobs-1);
            messages = combine(messages ,buffer);
        }
        i += j-1;
        if (chain_wrapped) break;
    }

    if (!SUMMARY_OUTPUT && !EBAI_OUTPUT){
        sprintf (buffer,"# * total number of chains found: %d\n", chains);
        messages = combine(messages , buffer);
    }

    /* If the number of chains is less than 2, the search for knots failed. */
    if (chains < 2)
        return -1;

    KNOTS = 4;
    *knots = malloc (KNOTS * sizeof (**knots));
    (*knots)[0] = data[chain[0].start].x; (*knots)[1] = data[chain[0].end].x;
    (*knots)[2] = data[chain[1].start].x; (*knots)[3] = data[chain[1].end].x;

    return 0;
}

double compute_flux (double phase, double *knots, double **ck)
{
    int j, k;
    double knot, dknot, flux;

    while (phase < -0.5) phase += 1.0;
    while (phase >  0.5) phase -= 1.0;

    if (phase < knots[0]) {
        j = KNOTS-1;
        knot = knots[j]-1.0;
        dknot = knots[0]-knots[KNOTS-1]+1.0;
    }
    else if (phase > knots[KNOTS-1]) {
        j = KNOTS-1;
        knot = knots[j];
        dknot = knots[0]-knots[KNOTS-1]+1.0;
    }
    else {
        j = 0;
        while (knots[j+1] < phase && j < KNOTS-1) j++;
        knot = knots[j];
        dknot = knots[j+1]-knots[j];
    }

    flux = ck[j][POLYORDER];
    for (k = 0; k < POLYORDER; k++) {
        flux += ck[j][k] * pow (phase-knot, POLYORDER-k);
        if (j == KNOTS-1 && k < POLYORDER-1)
            flux -= ck[j][k] * (phase-knot) * pow (dknot, POLYORDER-k-1);
    }

    return flux;
}

double polyfit (triplet *data, int nobs, int KNOTS, double *knots, int VERBOSE, int PRINT, double *outphase, double *outpolyfit)
{
    int i, j, k, intervals, kfinal, cum, int1index;
    double chisq, chi2tot, knot, pshift = 0.0;
    double **ck;

    int *n; /* Array of numbers of data points per interval */
    gsl_vector **x, **y, **w, **c;
    gsl_matrix **A, **cov;
    gsl_multifit_linear_workspace **mw;

    /* Count data points between knots: */
    n = calloc (KNOTS, sizeof (*n));

    /* Fast-forward to the first knot: */
    i = 0; cum = 0;
    while (data[i].x < knots[0]) { i++; cum++; }
    n[KNOTS-1] = i;
    int1index = cum;

    for (j = 0; j < KNOTS-1; j++) {
        while (data[i].x < knots[j+1] && i < nobs) i++;
        n[j] += i-cum; /* we need += because of the wrapped interval */
        if (n[j] <= POLYORDER) {
            /* The intervals between knots don't have enough points; bail out. */
            free (n);
            return 1e10;
        }
        cum += n[j];
    }

    /* Add post-last knot data to the last interval: */
    n[j] += nobs-cum;
    if (n[j] <= POLYORDER) {
        /* The intervals between knots don't have enough points; bail out. */
        free (n);
        return 1e10;
    }

    if (VERBOSE && !SUMMARY_OUTPUT && !EBAI_OUTPUT) {
        sprintf (buffer,"# Phase space partitioning:\n# \n");
        messages = combine(messages , buffer);
        for (k = 0; k < KNOTS-1; k++){
            sprintf (buffer, "#   interval %2d: [% 3.3lf, % 3.3lf), %3d data points\n", k, knots[k], knots[k+1], n[k]);
            messages = combine(messages ,buffer);
        }
        sprintf (buffer,"#   interval %2d: [% 3.3lf, % 3.3lf), %3d data points\n", k, knots[k], knots[0], n[k]);
        messages = combine(messages ,buffer);
    }

    /* Allocate arrays of vectors of interval data: */
    x = malloc (KNOTS * sizeof (*x));
    y = malloc (KNOTS * sizeof (*y));
    w = malloc (KNOTS * sizeof (*w));

    for (j = 0; j < KNOTS; j++) {
        x[j] = gsl_vector_alloc (n[j]);
        y[j] = gsl_vector_alloc (n[j]);
        w[j] = gsl_vector_alloc (n[j]);
    }

    /* Copy the (phase-shifted) data to these vectors: */
    cum = 0;
    for (j = 0; j < KNOTS-1; j++) {
        for (i = 0; i < n[j]; i++) {
            gsl_vector_set (x[j], i, data[int1index+cum+i].x - knots[j]);
            gsl_vector_set (y[j], i, data[int1index+cum+i].y);
            gsl_vector_set (w[j], i, data[int1index+cum+i].z);
            if (DEBUG)
                sprintf (buffer,"%d: %lf\t%lf\t%lf\n", j, data[int1index+cum+i].x - knots[j], data[int1index+cum+i].y, data[int1index+cum+i].z);
        }
        cum += n[j];
    }

    /* Copy the wrapped interval data to these vectors: */
    knot = knots[j];
    for (i = 0; i < n[j]; i++) {
        if (int1index+cum+i == nobs) {
            cum = -int1index-i;
            knot -= 1.0;
        }
        gsl_vector_set (x[j], i, data[int1index+cum+i].x - knot);
        gsl_vector_set (y[j], i, data[int1index+cum+i].y);
        gsl_vector_set (w[j], i, data[int1index+cum+i].z);
        if (DEBUG)
            sprintf (buffer,"%d: %lf\t%lf\t%lf\n", j, data[int1index+cum+i].x - knot, data[int1index+cum+i].y, data[int1index+cum+i].z);
    }

    /* Write out a header: */
    if (VERBOSE && !SUMMARY_OUTPUT && !EBAI_OUTPUT) {
        sprintf (buffer,"# \n# Weighted least-squares solution of the polyfit:\n# \n");
        messages = combine(messages ,buffer);
        sprintf (buffer,"#    knot\t\t");
        messages = combine(messages ,buffer);
        for (k = 0; k < POLYORDER+1; k++){
            sprintf (buffer,"   c[%d]\t\t", k);
            messages = combine(messages ,buffer);
        }
        sprintf (buffer,"\n");
        messages = combine(messages ,buffer);
    }

    /* If polynomial order is 1, the last interval is determined by constraints: */
    if (POLYORDER == 1) intervals = KNOTS-1; else intervals = KNOTS;

    /* Allocate all vectors and matrices: */
    A   = malloc (intervals * sizeof (*A));
    c   = malloc (intervals * sizeof (*c));
    cov = malloc (intervals * sizeof (*cov));
    mw  = malloc (intervals * sizeof (*mw));

    /* The first interval has all polynomial coefficients free. */
    A[0]   = gsl_matrix_alloc (n[0], POLYORDER+1);
    c[0]   = gsl_vector_alloc (POLYORDER+1);
    cov[0] = gsl_matrix_alloc (POLYORDER+1, POLYORDER+1);
    mw[0]  = gsl_multifit_linear_alloc (n[0], POLYORDER+1);

    /* Intervals 1...(N-1) are constrained by the connectivity constraint. */
    for (j = 1; j < KNOTS-1; j++) {
        A[j]   = gsl_matrix_alloc (n[j], POLYORDER);
        c[j]   = gsl_vector_alloc (POLYORDER);
        cov[j] = gsl_matrix_alloc (POLYORDER, POLYORDER);
        mw[j]  = gsl_multifit_linear_alloc (n[j], POLYORDER);
    }

    /* The last interval has two constraints (connectivity and periodicity). */
    if (j < intervals) {
        A[j]   = gsl_matrix_alloc (n[j], POLYORDER-1);
        c[j]   = gsl_vector_alloc (POLYORDER-1);
        cov[j] = gsl_matrix_alloc (POLYORDER-1, POLYORDER-1);
        mw[j]  = gsl_multifit_linear_alloc (n[j], POLYORDER-1);
    }

    /* Allocate polynomial coefficients: */
    ck = malloc (KNOTS * sizeof (*ck));
    for (k = 0; k < KNOTS; k++)
        ck[k] = malloc ((POLYORDER+1) * sizeof (**ck));

    /* Set all elements to all matrices: */
    for (j = 0; j < intervals; j++) {
        if      (j == 0)       kfinal = POLYORDER+1;
        else if (j == KNOTS-1) kfinal = POLYORDER-1;
        else                   kfinal = POLYORDER;
        for (i = 0; i < n[j]; i++)
            for (k = 0; k < kfinal; k++)
                gsl_matrix_set (A[j], i, k, pow (gsl_vector_get (x[j], i), POLYORDER-k));
    }

    /*********************   FITTING THE 1ST INTERVAL:   **********************/

    gsl_multifit_wlinear (A[0], w[0], y[0], c[0], cov[0], &chisq, mw[0]);

    for (k = 0; k < POLYORDER+1; k++)
        ck[0][k] = gsl_vector_get (c[0], k);
    chi2tot = chisq;

    if (VERBOSE && !SUMMARY_OUTPUT && !EBAI_OUTPUT) {
        sprintf (buffer,"# % 12.12lf\t", knots[0]);
        messages = combine(messages ,buffer);
        for (k = 0; k < POLYORDER+1; k++){
            sprintf (buffer,"% 12.12lf\t", ck[0][POLYORDER-k]);
            messages = combine(messages ,buffer);
        }
        sprintf (buffer,"\n");
        messages = combine(messages ,buffer);
    }

    /********************   FITTING INTERVALS 2-(N-1):   **********************/

    for (j = 1; j < KNOTS-1; j++) {
        /* Satisfy the connectivity constraint: */
        ck[j][POLYORDER] = ck[j-1][POLYORDER];
        for (k = 0; k < POLYORDER; k++)
            ck[j][POLYORDER] += ck[j-1][k] * pow (knots[j]-knots[j-1], POLYORDER-k);

        /* Apply the connectivity constraint: */
        for (i = 0; i < n[j]; i++)
            gsl_vector_set (y[j], i, gsl_vector_get (y[j], i) - ck[j][POLYORDER]);

        gsl_multifit_wlinear (A[j], w[j], y[j], c[j], cov[j], &chisq, mw[j]);
        chi2tot += chisq;

        for (k = 0; k < POLYORDER; k++)
            ck[j][k] = gsl_vector_get (c[j], k);

        if (VERBOSE && !SUMMARY_OUTPUT && !EBAI_OUTPUT) {
            sprintf (buffer,"# % 12.12lf\t", knots[j]);
            messages = combine(messages ,buffer);
            for (k = 0; k < POLYORDER+1; k++){
                sprintf (buffer,"% 12.12lf\t", ck[j][POLYORDER-k]);
                messages = combine(messages ,buffer);
            }
            sprintf (buffer,"\n");
            messages = combine(messages ,buffer);
        }
    }

    /********************   FITTING THE LAST INTERVAL:   **********************/

    /* Satisfy the connectivity constraint: */
    ck[j][POLYORDER] = ck[j-1][POLYORDER];
    for (k = 0; k < POLYORDER; k++)
        ck[j][POLYORDER] += ck[j-1][k] * pow (knots[j]-knots[j-1], POLYORDER-k);

    /* Satisfy the periodicity constraint: */
    ck[j][POLYORDER-1] = (ck[0][POLYORDER]-ck[j][POLYORDER])/(knots[0]-knots[j]+1.0);

    /* Apply both constraints: */
    for (i = 0; i < n[j]; i++)
        gsl_vector_set (y[j], i, gsl_vector_get (y[j], i) - ck[j][POLYORDER] - ck[j][POLYORDER-1] * gsl_vector_get (x[j], i));

    if (POLYORDER > 1) {
        for (k = 0; k < POLYORDER-1; k++)
            for (i = 0; i < n[j]; i++)
                gsl_matrix_set (A[j], i, k, gsl_matrix_get (A[j], i, k) - gsl_vector_get (x[j], i) * pow(knots[0]-knots[j]+1, POLYORDER-k-1));

        gsl_multifit_wlinear (A[j], w[j], y[j], c[j], cov[j], &chisq, mw[j]);

        for (k = 0; k < POLYORDER-1; k++)
            ck[j][k] = gsl_vector_get (c[j], k);
        chi2tot += chisq;
    }
    else {
        /*
         * If we are fitting linear polynomials, there's nothing to fit here
         * because both knots are constrained (connectivity and periodicity).
         * However, we still need to traverse data points to get chi2.
         */

        chisq = 0.0;
        for (i = 0; i < n[j]; i++)
            chisq += gsl_vector_get (w[j], i) * gsl_vector_get (y[j], i) * gsl_vector_get (y[j], i);
        chi2tot += chisq;
    }

    if (VERBOSE && !SUMMARY_OUTPUT && !EBAI_OUTPUT) {
        sprintf (buffer,"# % 12.12lf\t", knots[j]);
        messages = combine(messages ,buffer);
        for (k = 0; k < POLYORDER+1; k++){
            sprintf (buffer,"% 12.12lf\t", ck[j][POLYORDER-k]);
            messages = combine(messages ,buffer);
        }
        sprintf (buffer,"\n");
        messages = combine(messages ,buffer);
    }

    if (PRINT) {
        double phase;

        if (POLYORDER == 2) {
            double minval, maxval, xval, yval, xmax = 0.0;

            minval = 1e3;
            maxval = -1.0;

            for (j = 0; j < KNOTS; j++) {
                double yv1 = compute_flux (knots[j], knots, ck);
                double yv2 = compute_flux (j == KNOTS-1 ? knots[0] : knots[j+1], knots, ck);

                /* Concave parabola: check for minimum at end-points, find maximum. */
                if (ck[j][0] < 0) {
                    /* Check for minimum at end-points: */
                    if (yv1 < minval || yv2 < minval) {
                        pshift = yv1 < yv2 ? knots[j] : j == KNOTS-1 ? knots[0] : knots[j+1];
                        minval = yv1 < yv2 ? yv1 : yv2;
                    }

                    /* Find the maximum: */
                    xval = knots[j]-(ck[j][1]-(j == KNOTS-1 ? ck[j][0]*(knots[0]-knots[j]+1.0) : 0))/2.0/ck[j][0];

                    /* If the maximum is not enclosed in the interval, check end-points: */
                    if ((xval < knots[j] || xval > (j == KNOTS-1 ? knots[0]+1.0 : knots[j+1])) && (yv1 > maxval || yv2 > maxval)) {
                        xmax = yv1 > yv2 ? knots[j] : j == KNOTS-1 ? knots[0] : knots[j+1];
                        maxval = yv1 > yv2 ? yv1 : yv2;
                    }
                    /* If the maximum is enclosed in the interval: */
                    else {
                        yval = compute_flux (xval, knots, ck);
                        if (yval > maxval) {
                            xmax = xval;
                            maxval = yval;
                        }
                    }
                }
                /* Convex parabola: check for maximum at end-points, find minimum. */
                else {
                    if (yv1 > maxval || yv2 > maxval) {
                        xmax = yv1 > yv2 ? knots[j] : j == KNOTS-1 ? knots[0] : knots[j+1];
                        maxval = yv1 > yv2 ? yv1 : yv2;
                    }

                    /* Find the minimum: */
                    xval = knots[j]-(ck[j][1]-(j == KNOTS-1 ? ck[j][0]*(knots[0]-knots[j]+1.0) : 0))/2.0/ck[j][0];

                    /* If the minimum is not enclosed in the interval, check end-points: */
                    if ((xval < knots[j] || xval > (j == KNOTS-1 ? knots[0]+1.0 : knots[j+1])) && (yv1 < minval || yv2 < minval)) {
                        pshift = yv1 < yv2 ? knots[j] : j == KNOTS-1 ? knots[0] : knots[j+1];
                        minval = yv1 < yv2 ? yv1 : yv2;
                    }
                    /* If the minimum is enclosed in the interval: */
                    else {
                        yval = compute_flux (xval, knots, ck);
                        if (minval > yval) {
                            minval = yval;
                            pshift = xval;
                        }
                    }
                }
            }

            if (SUMMARY_OUTPUT)
                sprintf (buffer,"%lf\n", pshift);

            else if (!EBAI_OUTPUT) {
                sprintf (buffer,"# \n# Polyfit minimum occurs at: (%lf, %lf)\n", pshift, minval);
                messages = combine(messages ,buffer);
                sprintf (buffer,"# Polyfit maximum occurs at: (%lf, %lf)\n", xmax, maxval);
                messages = combine(messages ,buffer);
                sprintf (buffer,"# Polyfit amplitude is: %lf (%3.3lf%%)\n", maxval-minval, (maxval-minval)/maxval*100.0);
                messages = combine(messages ,buffer);
            }
        }

        if (!SUMMARY_OUTPUT && !EBAI_OUTPUT) {
            //sprintf (buffer,"# \n# Theoretical light curve:\n# \n#   Phase:\t   Flux:\n");
            for (i = 0; i <= NVERT; i++) {
                phase = -0.5 + (double) i / NVERT;
                if (APPLY_PSHIFT) phase += pshift;
                /*
                if (CSV_OUTPUT)
                    sprintf (buffer,"%lf,%12.12lf\n", phase, compute_flux (phase, knots, ck));
                else
                    sprintf (buffer,"  % lf\t% 12.12lf\n", phase, compute_flux (phase, knots, ck));
                */
                outphase[i]   = phase;
                outpolyfit[i] = compute_flux (phase, knots, ck);
            }
        }
        else if (EBAI_OUTPUT) {
            for (i = 0; i <= NVERT; i++) {
                phase = -0.5 + (double) i / NVERT;
                if (APPLY_PSHIFT) phase += pshift;
                sprintf (buffer,"% 12.12lf\n", compute_flux (phase, knots, ck));
            }
        }
    }

    /* Done! Wrap it up: */
    for (i = 0; i < KNOTS-1; i++) {
        gsl_vector_free (x[i]);
        gsl_vector_free (y[i]);
        gsl_vector_free (w[i]);
        free (ck[i]);
    }
    free (x);
    free (y);
    free (w);
    free (n);
    free (ck);

    for (i = 0; i < intervals; i++) {
        gsl_vector_free (c[i]);
        gsl_matrix_free (A[i]);
        gsl_matrix_free (cov[i]);
        gsl_multifit_linear_free (mw[i]);
    }
    free (c);
    free (A);
    free (cov);
    free (mw);

    return chi2tot;
}

double *tokenize(char *line, int *token_no)
{
    double *tokens = NULL;
    int LEN = 0;

    char *token;

    if (CSV_INPUT)
        token = strtok(line, ",");
    else
        token = strtok(line, " \t");

    if (!token)
        return NULL;

    while (token) {
        LEN += 1;
        tokens = realloc(tokens, LEN*sizeof(*tokens));
        tokens[LEN-1] = atof(token);
        if (CSV_INPUT)
            token = strtok(NULL, ",");
        else
            token = strtok(NULL, " \t");
    }

    *token_no = LEN;
    return tokens;
}

char * polyfitter (
  double* inphase,
  double* influx,
  double* inerror,
  int nobs,
  int order,
  int ITERS,
  double STEP_SIZE,
  int vertices,
  int nKNOTS,
  double* knots,
  int FIND_KNOTS,
  int FIND_STEP,
  int chainlength,
  int applypshift,
  double* outphase,
  double* outpolyfit
)
{
    int i, iter, status;
    triplet *data;
    //FILE *in;
    double chi2, chi2test, u;
    //char line[255];

    //double *knots = NULL;
    double knotsTest[4];
    double *test;
    //int colno, inputs = 0;
    //double *cols;

    gsl_rng *r;

    messages = "";

    CHAIN_LENGTH = chainlength;
    KNOTS = nKNOTS;
    POLYORDER = order;
    NVERT = vertices;
    APPLY_PSHIFT = applypshift;

    /********************   READ DATA:   **********************/
    /* Read the input data: */
    data = malloc (nobs*sizeof(*data));

    for(i=0;i < nobs; i++){
        data[i].x = inphase[i];
        data[i].y = influx[i];
        data[i].z = inerror[i];
    }

    /*
    if (argc < 2) {
        sprintf (buffer,"polyfit %s\n\n", VERSION);
        sprintf (buffer,"Usage: ./polyfit [options] lc.dat\n\n");
        sprintf (buffer,"File lc.dat should have at least 1 column (flux),\n");
        sprintf (buffer,"2 columns (phase and flux), or 3 columns (phase,\n");
        sprintf (buffer,"flux, and standard deviation)\n\n");
        sprintf (buffer,"Options:\n\n");
        sprintf (buffer,"  -o order          ..  fitting polynomial order (default: 2)\n");
        sprintf (buffer,"  -i iters          ..  number of iterations (default: 10000)\n");
        sprintf (buffer,"  -s step           ..  step for random knot displacement (default: 0.01)\n");
        sprintf (buffer,"  -k k1 k2 ... kN   ..  explicit list of knots\n");
        sprintf (buffer,"  -n vertices       ..  number of vertices in the computed fit\n");
        sprintf (buffer,"  -c c1 c2 [c3]     ..  input file columns to be parsed (starting from 0)\n");
        sprintf (buffer,"  --find-knots      ..  attempt to find knots automatically\n");
        sprintf (buffer,"  --find-step       ..  attempt to find step automatically\n");
        sprintf (buffer,"  --chain-length    ..  minimum chain length for automatic knot search\n");
        sprintf (buffer,"  --apply-pshift    ..  shift phase so that the polyfit minimum is at phase 0\n");
        sprintf (buffer,"  --csv-input       ..  take comma-separated-value file as input\n");
        sprintf (buffer,"  --csv-output      ..  print comma-separated-values at output\n");
        sprintf (buffer,"  --limits          ..  set lower/upper phase limit (detault: no filtering)\n");
        sprintf (buffer,"  --ebai-output     ..  write a one-column output in ebai-compatible format\n");
        sprintf (buffer,"  --summary-output  ..  output a single summary line from polyfit\n\n");
        exit (0);
    }

    for (i = 1; i < argc; i++) {
        if (strcmp (argv[i], "-o") == 0)
            POLYORDER = atoi (argv[++i]);
        if (strcmp (argv[i], "-i") == 0)
            ITERS = atoi (argv[++i]);
        if (strcmp (argv[i], "-s") == 0)
            STEP_SIZE = atof (argv[++i]);
        if (strcmp (argv[i], "-n") == 0)
            NVERT = atoi (argv[++i]);
        if (strcmp (argv[i], "-k") == 0) {
            double knot;
            i++;
            KNOTS = 0;
            while (sscanf (argv[i], "%lf", &knot) == 1) {
                if (i == argc-1) break;
                KNOTS++;
                knots = realloc (knots, KNOTS * sizeof (*knots));
                knots[KNOTS-1] = knot;
                i++;
            }
            i--;
        }
        if (strcmp (argv[i], "-c") == 0) {
            int val;

            i++;
            inputs = 0;
            while (sscanf (argv[i], "%d", &val) == 1) {
                if (i == argc-1) break;
                COLIDX[inputs] = val;
                inputs++;
                i++;
            }
            i--;
        }
        if (strcmp(argv[i], "--limits") == 0) {
            sscanf(argv[i+1], "%lf", &LLIMIT);
            sscanf(argv[i+2], "%lf", &ULIMIT);
            i += 2;
        }
        if (strcmp (argv[i], "--find-knots") == 0)
            FIND_KNOTS = 1;
        if (strcmp (argv[i], "--find-step") == 0)
            FIND_STEP = 1;
        if (strcmp (argv[i], "--chain-length") == 0)
            CHAIN_LENGTH = atoi (argv[++i]);
        if (strcmp (argv[i], "--csv-input") == 0)
            CSV_INPUT = 1;
        if (strcmp (argv[i], "--csv-output") == 0)
            CSV_OUTPUT = 1;
        if (strcmp (argv[i], "--summary-output") == 0)
            SUMMARY_OUTPUT = 1;
        if (strcmp (argv[i], "--ebai-output") == 0)
            EBAI_OUTPUT = 1;
        if (strcmp (argv[i], "--apply-pshift") == 0)
            APPLY_PSHIFT = 1;
    }

    if (SUMMARY_OUTPUT)
        sprintf (buffer,"%s\t", argv[argc-1]);

    in = fopen (argv[argc-1], "r");
    if (!in) {
        sprintf (buffer,"file %s not found, aborting.\n", argv[argc-1]);
        exit (0);
    }
    */

    /********************   READ DATA:   **********************/
    /* Read the input data: */
    /*
    data = malloc (CHUNK*sizeof(*data));
    i = 0;
    while (!feof (in)) {
        if (!fgets (line, 255, in)) break;
        if (line[0] == '\n') break;       // break on empty lines
        if (line[0] == '#') continue;     // ignore comments

        // Break up the line into tokens:
        cols = tokenize(line, &colno);

        // Guess the role of columns if not given by the -c switch:
        if (i == 0 && COLIDX[0] == -1 && COLIDX[1] == -1 && COLIDX[2] == -1) {
            if (colno == 1) {
                COLIDX[1] = 0;
            }
            else if (colno == 2) {
                COLIDX[0] = 0;
                COLIDX[1] = 1;
            }
            else {
                COLIDX[0] = 0;
                COLIDX[1] = 1;
                COLIDX[2] = 2;
            }
        }

        inputs = colno <= 3 ? colno : 3;

        if (colno <= COLIDX[0] || colno <= COLIDX[1] || colno <= COLIDX[2]) {
            sprintf(buffer,"The input file has %d columns, but column index %d is requested; aborting.\n", colno, COLIDX[2] <= colno ? COLIDX[2] : (COLIDX[1] <= colno ? COLIDX[1] : COLIDX[0]));
            exit(0);
        }

        // if (i == 0)
        //     sprintf(buffer,"[0]=%d [1]=%d [2]=%d\n", COLIDX[0], COLIDX[1], COLIDX[2]);

        // Check if the data point is within limits:
        if ((COLIDX[0] != -1) && (cols[COLIDX[0]] < LLIMIT || cols[COLIDX[0]] > ULIMIT))
            continue;

        if (inputs == 1) {
            data[i].y = cols[COLIDX[1]];
            data[i].z = 1e-3;
        }
        else if (inputs == 2) {
            data[i].x = cols[COLIDX[0]];
            data[i].y = cols[COLIDX[1]];
            data[i].z = 1e-3;
        }
        else {
            data[i].x = cols[COLIDX[0]];
            data[i].y = cols[COLIDX[1]];
            data[i].z = cols[COLIDX[2]];
        }

        free(cols);

        i++;
        if (i % CHUNK == 0)
            data = realloc(data, (i/CHUNK+1)*(CHUNK)*sizeof(*data));
    }
    fclose (in);
    nobs = i;
    */

    /*
    if (!SUMMARY_OUTPUT && !EBAI_OUTPUT)
        sprintf (buffer,"# %d data points read in from %s.\n", nobs, argv[argc-1]);

    // If only fluxes are provided, add equidistant nobs phases:
    if (inputs == 1) {
        if (!SUMMARY_OUTPUT && !EBAI_OUTPUT)
            sprintf(buffer,"# * no phases given, assuming equidistant sampling between -0.5 and 0.5.\n");
        for (i=0; i<nobs; i++) {
            data[i].x = -0.5 + (double) i/(nobs-1);
        }
    }
    */


    /* Sort data in phase: */
    qsort (data, nobs, sizeof (*data), sort_by_phase);

    /********************   FIND KNOTS:   **********************/
    if (FIND_KNOTS) {
        status = find_knots (data, nobs, &knots);
        if (!SUMMARY_OUTPUT && !EBAI_OUTPUT) {
            if (status != 0){
                sprintf (buffer,"# * automatic search for knots failed, reverting to defaults.\n");
                messages = combine(messages ,buffer);
            }
            else{
                sprintf (buffer,"# * automatic search for knots successful.\n");
                messages = combine(messages ,buffer);
            }
            sprintf (buffer,"# \n");
            messages = combine(messages ,buffer);
        }
    }

    /* The initial phase intervals for knots: */
    if (!knots) {
        KNOTS = 4;
        knots = malloc (KNOTS * sizeof (*knots));
        knots[0] = -0.4; knots[1] = -0.1; knots[2] = 0.1; knots[3] = 0.4;
    }

    /* Sort the knots in ascending order: */
    qsort (knots, KNOTS, sizeof (*knots), sort_by_value);

    /********************   FIND STEP:   **********************/
    if (FIND_STEP) {
        double diff;
        /* Step size would be the minimum width between two knots / 5: */
        diff = fabs (knots[1]-knots[0]);
        for (i = 1; i < KNOTS-1; i++)
            if (fabs (knots[i+1]-knots[i]) < diff)
                diff = fabs (knots[i+1]-knots[i]);
        if (fabs (knots[i+1]-knots[0]) < diff)
            diff = fabs (knots[i+1]-knots[0]);

        STEP_SIZE = diff / 5.0;
    }

    /****************************************************************/
    /*********    This part corrects wrong initial KNOTS       ******/
    /********* if KNOTS search fails use fixed inital values   ******/
    /****************************************************************/

    /********* Secondary is inside primary:   ************/
    /* If knots are all around primary minimum aka secondary is inside primary. -- narrow primary minimum */
    if ( ((knots[KNOTS-1]-knots[0])>0.001) && ((knots[KNOTS-1]-knots[0])<0.49) && ((knots[2]-knots[1])<0.015) ){
        KNOTS = 4;
        knots = malloc (KNOTS * sizeof (*knots));
        knots[0] = -0.4; knots[1] = -0.1; knots[2] = 0.1; knots[3] = 0.4;
    }
    /* If knots are all around primary minimum aka secondary is inside primary. -- wide primary minimum */
    if ( ((knots[KNOTS-1]-knots[0])>0.2) && ((knots[KNOTS-1]-knots[0])<0.49) && ((knots[2]-knots[1])<0.15) ){
        KNOTS = 4;
        knots = malloc (KNOTS * sizeof (*knots));
        knots[0] = -0.4; knots[1] = -0.1; knots[2] = 0.1; knots[3] = 0.4;
    }

    /********* If knots are too close:   ************/
    if ( (knots[KNOTS-1]-knots[0])<0.015 ){
        KNOTS = 4;
        knots = malloc (KNOTS * sizeof (*knots));
        knots[0] = -0.4; knots[1] = -0.1; knots[2] = 0.1; knots[3] = 0.4;
    }

    /********* Secondary is inside primary, which is around -0.5 ??? 0.5 ************/
    /* Check knots again but handle if minimum is at phase 0.5 (aka around -0.5 and 0.5) */
    for(i=0; i<KNOTS; i++){
        if (knots[i]<0) { knotsTest[i] = knots[i] + 0.5; }
        else {knotsTest[i] = knots[i] - 0.5;}
    }

    /* sort shifted knosts */
    qsort (knotsTest, KNOTS, sizeof (*knotsTest), sort_by_value);

    /* If knots are all around primary minimum aka secondary is inside primary. -- narrow primary minimum */
    if ( ((knotsTest[KNOTS-1]-knotsTest[0])>0.001) && ((knotsTest[KNOTS-1]-knotsTest[0])<0.49) && ((knotsTest[2]-knotsTest[1])<0.069) ){
        KNOTS = 4;
        knots = malloc (KNOTS * sizeof (*knotsTest));
        knots[0] = -0.4; knots[1] = -0.1; knots[2] = 0.1; knots[3] = 0.4;
    }
    /* If knots are all around primary minimum aka secondary is inside primary. -- wide primary minimum */
    if ( ((knotsTest[KNOTS-1]-knotsTest[0])>0.2) && ((knotsTest[KNOTS-1]-knotsTest[0])<0.49) && ((knotsTest[2]-knotsTest[1])<0.15) ){
        KNOTS = 4;
        knots = malloc (KNOTS * sizeof (*knots));
        knots[0] = -0.4; knots[1] = -0.1; knots[2] = 0.1; knots[3] = 0.4;
    }

    /********* Secondary minimum is a sharp peak:   ************/
    /* If secondary minimum is sharp, fix secondary's knots. -- sharp minimum is left to primary */
    if ( ((knots[1]-knots[0])<0.015) && ((knots[3]-knots[2])>0.05) ){
        knots[0] = -0.4; knots[1] = 0.4;
    }
    /* If secondary minimum is sharp, fix secondary's knots. -- sharp minimum is right to primary */
    if ( ((knots[1]-knots[0])>0.05) && ((knots[3]-knots[2])<0.015) ){
        knots[2] = -0.4; knots[3] = 0.4;
    }

    /********* Secondary minimum is a sharp peak, around phase -0.5 ??? 0.5  ************/
    /* Check knots again but handle if sharp minimum is at phase 0.5 (aka around -0.5 and 0.5) */
    for(i=0; i<KNOTS; i++){
        if (knots[i]<0.25) { knotsTest[i] = knots[i] + 0.25; }
        else { knotsTest[i] = knots[i] - 0.75; }
    }

    /* sort shifted knosts */
    qsort (knotsTest, KNOTS, sizeof (*knotsTest), sort_by_value);

    /* If secondary minimum is sharp, fix secondary's knots. -- sharp minimum is left to primary */
    if ( ((knotsTest[1]-knotsTest[0])<0.005) && ((knotsTest[3]-knotsTest[2])>0.1) ){
        knots[0] = -0.4; knots[1] = 0.4;
    }
    /* If secondary minimum is sharp, fix secondary's knots. -- sharp minimum is right to primary */
    if ( ((knotsTest[1]-knotsTest[0])>0.1) && ((knotsTest[3]-knotsTest[2])<0.005) ){
        knots[2] = -0.4; knots[3] = 0.4;
    }

    /* Sort the knots in ascending order: */
    qsort (knots, KNOTS, sizeof (*knots), sort_by_value);

    /********************   PRINT INFORMATION:   **********************/
    if (!SUMMARY_OUTPUT && !EBAI_OUTPUT) {
        sprintf (buffer,"# Fitting polynomial order: %d\n", POLYORDER);
        messages = combine( messages, buffer );
        sprintf (buffer,"# Initial set of knots: {");
        messages = combine( messages, buffer );
        for (i = 0; i < KNOTS-1; i++){
            sprintf (buffer,"%lf, ", knots[i]);
            messages = combine(messages ,buffer);
        }
        sprintf (buffer,"%lf}\n", knots[i]);
        messages = combine(messages ,buffer);
        sprintf (buffer,"# Number of iterations for knot search: %d\n", ITERS);
        messages = combine(messages ,buffer);
        sprintf (buffer,"# Step size for knot search: %lf\n# \n", STEP_SIZE);
        messages = combine(messages ,buffer);
    }

    /********* If primary minimum is a sharp peak near phase 0  ************/
    /* Sort absolute values -> if there is peak near 0 and 2nd peak far from 0 */
    for(i=0; i<KNOTS; i++){
        knotsTest[i] = fabs(knots[i]);
    }

    /* sort shifted knosts */
    qsort (knotsTest, KNOTS, sizeof (*knotsTest), sort_by_value);

    if ( (knotsTest[0] < 0.1) && ( (knotsTest[1]-knotsTest[0]) > 0.1) ){
        KNOTS = 4;
        knots = malloc (KNOTS * sizeof (*knotsTest));
        knots[0] = -0.4; knots[1] = -0.1; knots[2] = 0.1; knots[3] = 0.4;
    }

    /********* If both minima are very thin -> use 4th order fit  ************/
    /* If there is NO minimum at phase -0.5 - 0.5 */
    /*
    if ( ((knots[1]-knots[0])<0.06) && ((knots[3]-knots[2])<0.06) ){
        POLYORDER = 4;
    }
    // If one minimum is at phase -0.5 - 0.5
    for(i=0; i<KNOTS; i++){
        if (knots[i]<0.25) { knotsTest[i] = knots[i] + 0.25; }
        else { knotsTest[i] = knots[i] - 0.75; }
    }

    // sort shifted knosts
    qsort (knotsTest, KNOTS, sizeof (*knotsTest), sort_by_value);

    // If secondary minimum is sharp, fix secondary's knots. -- sharp minimum is left to primary
    if ( ((knotsTest[1]-knotsTest[0])<0.06) && ((knotsTest[3]-knotsTest[2])<0.06) ){
        POLYORDER = 4;
    }
    */

    /********************************************************************************/
    /********************   MCMC like search for best KNOTS:   **********************/
    /********************************************************************************/

    r = gsl_rng_alloc (gsl_rng_mt19937);
    gsl_rng_set (r, 1);

    chi2 = polyfit (data, nobs, KNOTS, knots, 0, 0, outphase, outpolyfit);
    if (!SUMMARY_OUTPUT && !EBAI_OUTPUT){
        sprintf (buffer,"# Original chi2: %e\n", chi2);
        messages = combine(messages ,buffer);
    }

    test = malloc (KNOTS * sizeof (*test));

    /*
    for (iter = 0; iter < ITERS; iter++) {
        for (i = 0; i < KNOTS; i++) {
            u = gsl_rng_uniform (r);
            test[i] = knots[i] + STEP_SIZE * 2 * u - STEP_SIZE;
            if (test[i] < -0.5) test[i] += 1.0;
            if (test[i] >  0.5) test[i] -= 1.0;
        }

        qsort (test, KNOTS, sizeof (*test), sort_by_value);

        chi2test = polyfit (data, nobs, KNOTS, test, 0, 0);
        if (chi2test < chi2) {
            chi2 = chi2test;
            for (i = 0; i < KNOTS; i++)
                knots[i] = test[i];
        }
    }
    */

    double chi2arr[200];
    double chi2med, chi2MAD;
    double work[200];
    double chi2stored;
    iter = 0;
    while (iter < ITERS) {
        for (i = 0; i < KNOTS; i++) {
            u = gsl_rng_uniform (r);
            test[i] = knots[i] + STEP_SIZE * 2 * u - STEP_SIZE;
            if (test[i] < -0.5) test[i] += 1.0;
            if (test[i] >  0.5) test[i] -= 1.0;
        }

        qsort (test, KNOTS, sizeof (*test), sort_by_value);

        chi2test = polyfit (data, nobs, KNOTS, test, 0, 0, outphase, outpolyfit);
        chi2stored = chi2;
        if (chi2test < chi2) {
            chi2 = chi2test;
            for (i = 0; i < KNOTS; i++)
                knots[i] = test[i];
        }
        if (iter<200){
            chi2arr[iter] = chi2test;
        }
        else{
            if (iter==200){
                chi2med = gsl_stats_median(chi2arr, 1, 200);
                chi2MAD = gsl_stats_mad(chi2arr, 1, 200, work);
                sprintf(buffer,"# Chi2med=%f chi2MAD=%f Threshold=%f\n",chi2med,chi2MAD,chi2med - 5*chi2MAD);
                messages = combine(messages ,buffer);
            }
            if ((chi2test < chi2stored) && (chi2test < chi2med - 5*chi2MAD)){
                sprintf(buffer,"# BREAKING @%d\n",iter);
                messages = combine(messages ,buffer);
                break;
            }
        }

        iter++;
    }

    if (!SUMMARY_OUTPUT && !EBAI_OUTPUT){
        sprintf (buffer,"# Final chi2:    %e\n", chi2);
        messages = combine(messages ,buffer);
    }

    if (SUMMARY_OUTPUT)
        sprintf (buffer,"%lf\t", chi2);

    chi2 = polyfit (data, nobs, KNOTS, knots, 1, 1, outphase, outpolyfit);

    gsl_rng_free (r);
    free (data);
    free (test);
    //free (knots);

    return messages;
    }
