#ifndef VALID_H
#define VALID_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"

INT lastValidHead = 0;
INT lastValidTail = 0;
INT lastValidRel = 0;

REAL l1_valid_filter_tot = 0, l1_valid_tot = 0, r1_valid_tot = 0, r1_valid_filter_tot = 0, l_valid_tot = 0, r_valid_tot = 0, l_valid_filter_rank = 0, l_valid_rank = 0, l_valid_filter_reci_rank = 0, l_valid_reci_rank = 0;
REAL l3_valid_filter_tot = 0, l3_valid_tot = 0, r3_valid_tot = 0, r3_valid_filter_tot = 0, l_valid_filter_tot = 0, r_valid_filter_tot = 0, r_valid_filter_rank = 0, r_valid_rank = 0, r_valid_filter_reci_rank = 0, r_valid_reci_rank = 0;
REAL rel3_valid_tot = 0, rel3_valid_filter_tot = 0, rel_valid_filter_tot = 0, rel_valid_filter_rank = 0, rel_valid_rank = 0, rel_valid_filter_reci_rank = 0, rel_valid_reci_rank = 0, rel_valid_tot = 0, rel1_valid_tot = 0, rel1_valid_filter_tot = 0;

REAL l1_valid_filter_tot_constrain = 0, l1_valid_tot_constrain = 0, r1_valid_tot_constrain = 0, r1_valid_filter_tot_constrain = 0, l_valid_tot_constrain = 0, r_valid_tot_constrain = 0, l_valid_filter_rank_constrain = 0, l_valid_rank_constrain = 0, l_valid_filter_reci_rank_constrain = 0, l_valid_reci_rank_constrain = 0;

REAL l3_valid_filter_tot_constrain = 0, l3_valid_tot_constrain = 0, r3_valid_tot_constrain = 0, r3_valid_filter_tot_constrain = 0, l_valid_filter_tot_constrain = 0, r_valid_filter_tot_constrain = 0, r_valid_filter_rank_constrain = 0, r_valid_rank_constrain = 0, r_valid_filter_reci_rank_constrain = 0, r_valid_reci_rank_constrain = 0;

REAL validHit1, validHit3, validHit10, validMr, validMrr;
REAL validHit1TC, validHit3TC, validHit10TC, validMrTC, validMrrTC;

extern "C" void initValid()
/*
 * Function: initTest
 * ----------------------------
 *   Khởi tạo giá trị kết quả test
 */
{
    lastValidHead = 0;
    lastValidTail = 0;
    lastValidRel = 0;
    l1_valid_filter_tot = 0, l1_valid_tot = 0, r1_valid_tot = 0, r1_valid_filter_tot = 0, l_valid_tot = 0, r_valid_tot = 0, l_valid_filter_rank = 0, l_valid_rank = 0, l_valid_filter_reci_rank = 0, l_valid_reci_rank = 0;
    l3_valid_filter_tot = 0, l3_valid_tot = 0, r3_valid_tot = 0, r3_valid_filter_tot = 0, l_valid_filter_tot = 0, r_valid_filter_tot = 0, r_valid_filter_rank = 0, r_valid_rank = 0, r_valid_filter_reci_rank = 0, r_valid_reci_rank = 0;
    REAL rel3_valid_tot = 0, rel3_valid_filter_tot = 0, rel_valid_filter_tot = 0, rel_valid_filter_rank = 0, rel_valid_rank = 0, rel_valid_filter_reci_rank = 0, rel_valid_reci_rank = 0, rel_valid_tot = 0, rel1_valid_tot = 0, rel1_valid_filter_tot = 0;

    l1_valid_filter_tot_constrain = 0, l1_valid_tot_constrain = 0, r1_valid_tot_constrain = 0, r1_valid_filter_tot_constrain = 0, l_valid_tot_constrain = 0, r_valid_tot_constrain = 0, l_valid_filter_rank_constrain = 0, l_valid_rank_constrain = 0, l_valid_filter_reci_rank_constrain = 0, l_valid_reci_rank_constrain = 0;
    l3_valid_filter_tot_constrain = 0, l3_valid_tot_constrain = 0, r3_valid_tot_constrain = 0, r3_valid_filter_tot_constrain = 0, l_valid_filter_tot_constrain = 0, r_valid_filter_tot_constrain = 0, r_valid_filter_rank_constrain = 0, r_valid_rank_constrain = 0, r_valid_filter_reci_rank_constrain = 0, r_valid_reci_rank_constrain = 0;
}

extern "C" void getValidHeadBatch(INT *ph, INT *pt, INT *pr)
/*
 * Function: getValidHeadBatch
 * ----------------------------
 *   Trả về validation head batch
 *
 *   ph: con trỏ head
 *   pt: con trỏ tail
 *   pr: con trỏ relation
 *
 */
{
    for (INT i = 0; i < entityTotal; i++)
    {
        ph[i] = i;
        pt[i] = validList[lastValidHead].t;
        pr[i] = validList[lastValidHead].r;
    }
    lastValidHead++;
}

extern "C" void getValidTailBatch(INT *ph, INT *pt, INT *pr)
/*
 * Function: getValidTailBatch
 * ----------------------------
 *   Trả về validation tail batch
 *
 *   ph: con trỏ head
 *   pt: con trỏ tail
 *   pr: con trỏ relation
 *
 */
{
    for (INT i = 0; i < entityTotal; i++)
    {
        ph[i] = validList[lastValidTail].h;
        pt[i] = i;
        pr[i] = validList[lastValidTail].r;
    }
    lastValidTail++;
}

extern "C" void getValidRelBatch(INT *ph, INT *pt, INT *pr)
/*
 * Function: getValidRelBatch
 * ----------------------------
 *   Trả về validation rel batch
 *
 *   ph: con trỏ head quản lý mảng ph
 *   pt: con trỏ tail quản lý mảng pt
 *   pr: con trỏ relation quản lý mảng pr
 *
 */
{
    for (INT i = 0; i < relationTotal; i++)
    {
        ph[i] = validList[lastValidRel].h;
        pt[i] = validList[lastValidRel].t;
        pr[i] = i;
    }
}

extern "C" void validHead(REAL *con, INT lastValidHead, bool type_constrain = false)
/*
 * Function: testHead
 * ----------------------------
 *   Trả về kết quả hits@10, hits@3, hit@1, mean rank, mean reciprocal rank (có type constraint)
 *
 *   *con:
 *   lastHead:
 *   type_constrain:
 *
 */
{
    INT h = validList[lastValidHead].h;
    INT t = validList[lastValidHead].t;
    INT r = validList[lastValidHead].r;
    INT lef, rig;
    // Kiểm tra điều kiện có dùng type constraint hay không?
    if (type_constrain)
    {
        lef = head_lef[r];
        rig = head_rig[r];
    }

    REAL minimal = con[h];
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;
    INT l_filter_s_constrain = 0;

    for (INT j = 0; j < entityTotal; j++)
    {
        if (j != h)
        {
            REAL value = con[j];
            if (value < minimal)
            {
                l_s += 1;
                if (not _find(j, t, r))
                    l_filter_s += 1;
            }
            // Nếu có dùng type constraint
            if (type_constrain)
            {
                while (lef < rig && head_type[lef] < j)
                    lef++;
                if (lef < rig && j == head_type[lef])
                {
                    if (value < minimal)
                    {
                        l_s_constrain += 1;
                        if (not _find(j, t, r))
                        {
                            l_filter_s_constrain += 1;
                        }
                    }
                }
            }
        }
    }

    // Tính toán hits@10, hits@3, hit@1, mean rank, mean reciprocal rank
    if (l_filter_s < 10)
        l_valid_filter_tot += 1;
    if (l_s < 10)
        l_valid_tot += 1;
    if (l_filter_s < 3)
        l3_valid_filter_tot += 1;
    if (l_s < 3)
        l3_valid_tot += 1;
    if (l_filter_s < 1)
        l1_valid_filter_tot += 1;
    if (l_s < 1)
        l1_valid_tot += 1;

    l_valid_filter_rank += (l_filter_s + 1);
    l_valid_rank += (1 + l_s);
    l_valid_filter_reci_rank += 1.0 / (l_filter_s + 1);
    l_valid_reci_rank += 1.0 / (l_s + 1);

    // Tính toán hits@10, hits@3, hit@1, mean rank, mean reciprocal rank với type constraint
    if (type_constrain)
    {
        if (l_filter_s_constrain < 10)
            l_valid_filter_tot_constrain += 1;
        if (l_s_constrain < 10)
            l_valid_tot_constrain += 1;
        if (l_filter_s_constrain < 3)
            l3_valid_filter_tot_constrain += 1;
        if (l_s_constrain < 3)
            l3_valid_tot_constrain += 1;
        if (l_filter_s_constrain < 1)
            l1_valid_filter_tot_constrain += 1;
        if (l_s_constrain < 1)
            l1_valid_tot_constrain += 1;

        l_valid_filter_rank_constrain += (l_filter_s_constrain + 1);
        l_valid_rank_constrain += (1 + l_s_constrain);
        l_valid_filter_reci_rank_constrain += 1.0 / (l_filter_s_constrain + 1);
        l_valid_reci_rank_constrain += 1.0 / (l_s_constrain + 1);
    }
}

extern "C" void validTail(REAL *con, INT lastValidTail, bool type_constrain = false)
/*
 * Function: testTail
 * ----------------------------
 *   Trả về kết quả hits@10, hits@3, hit@1, mean rank, mean reciprocal rank (có type constraint)
 *
 *   *con:
 *   lastTail:
 *   type_constrain:
 *
 */
{
    INT h = validList[lastValidTail].h;
    INT t = validList[lastValidTail].t;
    INT r = validList[lastValidTail].r;

    INT lef, rig;

    if (type_constrain)
    {
        lef = tail_lef[r];
        rig = tail_rig[r];
    }

    REAL minimal = con[t];

    INT r_s = 0;
    INT r_filter_s = 0;
    INT r_s_constrain = 0;
    INT r_filter_s_constrain = 0;

    for (INT j = 0; j < entityTotal; j++)
    {
        if (j != t)
        {
            REAL value = con[j];
            if (value < minimal)
            {
                r_s += 1;
                if (not _find(h, j, r))
                    r_filter_s += 1;
            }
            if (type_constrain)
            {
                while (lef < rig && tail_type[lef] < j)
                    lef++;
                if (lef < rig && j == tail_type[lef])
                {
                    if (value < minimal)
                    {
                        r_s_constrain += 1;
                        if (not _find(h, j, r))
                        {
                            r_filter_s_constrain += 1;
                        }
                    }
                }
            }
        }
    }

    if (r_filter_s < 10)
        r_valid_filter_tot += 1;
    if (r_s < 10)
        r_valid_tot += 1;
    if (r_filter_s < 3)
        r3_valid_filter_tot += 1;
    if (r_s < 3)
        r3_valid_tot += 1;
    if (r_filter_s < 1)
        r1_valid_filter_tot += 1;
    if (r_s < 1)
        r1_valid_tot += 1;

    r_valid_filter_rank += (1 + r_filter_s);
    r_valid_rank += (1 + r_s);
    r_valid_filter_reci_rank += 1.0 / (1 + r_filter_s);
    r_valid_reci_rank += 1.0 / (1 + r_s);

    if (type_constrain)
    {
        if (r_filter_s_constrain < 10)
            r_valid_filter_tot_constrain += 1;
        if (r_s_constrain < 10)
            r_valid_tot_constrain += 1;
        if (r_filter_s_constrain < 3)
            r3_valid_filter_tot_constrain += 1;
        if (r_s_constrain < 3)
            r3_valid_tot_constrain += 1;
        if (r_filter_s_constrain < 1)
            r1_valid_filter_tot_constrain += 1;
        if (r_s_constrain < 1)
            r1_valid_tot_constrain += 1;

        r_valid_filter_rank_constrain += (1 + r_filter_s_constrain);
        r_valid_rank_constrain += (1 + r_s_constrain);
        r_valid_filter_reci_rank_constrain += 1.0 / (1 + r_filter_s_constrain);
        r_valid_reci_rank_constrain += 1.0 / (1 + r_s_constrain);
    }
}

extern "C" void validRel(REAL *con)
/*
 * Function: testHead
 * ----------------------------
 *   Trả về kết quả hits@10, hits@3, hit@1, mean rank, mean reciprocal rank (có type constraint)
 *
 *   *con:
 *
 */
{
    INT h = validList[lastValidRel].h;
    INT t = validList[lastValidRel].t;
    INT r = validList[lastValidRel].r;

    REAL minimal = con[r];
    INT rel_s = 0;
    INT rel_filter_s = 0;

    for (INT j = 0; j < relationTotal; j++)
    {
        if (j != r)
        {
            REAL value = con[j];
            if (value < minimal)
            {
                rel_s += 1;
                if (not _find(h, t, j))
                    rel_filter_s += 1;
            }
        }
    }

    if (rel_filter_s < 10)
        rel_valid_filter_tot += 1;
    if (rel_s < 10)
        rel_valid_tot += 1;
    if (rel_filter_s < 3)
        rel3_valid_filter_tot += 1;
    if (rel_s < 3)
        rel3_valid_tot += 1;
    if (rel_filter_s < 1)
        rel1_valid_filter_tot += 1;
    if (rel_s < 1)
        rel1_valid_tot += 1;

    rel_valid_filter_rank += (rel_filter_s + 1);
    rel_valid_rank += (1 + rel_s);
    rel_valid_filter_reci_rank += 1.0 / (rel_filter_s + 1);
    rel_valid_reci_rank += 1.0 / (rel_s + 1);

    lastValidRel++;
}

extern "C" void valid_link_prediction(bool type_constrain = false)
/*
 * Function: test_link_prediction
 * ----------------------------
 *   In kết quả hits@10, hits@3, hit@1, mean rank, mean reciprocal rank (có type constraint)
 *
 *   type_constrain:
 *
 */
{
    l_valid_rank /= validTotal;
    r_valid_rank /= validTotal;
    l_valid_reci_rank /= validTotal;
    r_valid_reci_rank /= validTotal;

    l_valid_tot /= validTotal;
    l3_valid_tot /= validTotal;
    l1_valid_tot /= validTotal;

    r_valid_tot /= validTotal;
    r3_valid_tot /= validTotal;
    r1_valid_tot /= validTotal;

    // with filter
    l_valid_filter_rank /= validTotal;
    r_valid_filter_rank /= validTotal;
    l_valid_filter_reci_rank /= validTotal;
    r_valid_filter_reci_rank /= validTotal;

    l_valid_filter_tot /= validTotal;
    l3_valid_filter_tot /= validTotal;
    l1_valid_filter_tot /= validTotal;

    r_valid_filter_tot /= validTotal;
    r3_valid_filter_tot /= validTotal;
    r1_valid_filter_tot /= validTotal;

    printf("no type constraint results:\n");

    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_valid_reci_rank, l_valid_rank, l_valid_tot, l3_valid_tot, l1_valid_tot);
    printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_valid_reci_rank, r_valid_rank, r_valid_tot, r3_valid_tot, r1_valid_tot);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
           (l_valid_reci_rank + r_valid_reci_rank) / 2, (l_valid_rank + r_valid_rank) / 2, (l_valid_tot + r_valid_tot) / 2, (l3_valid_tot + r3_valid_tot) / 2, (l1_valid_tot + r1_valid_tot) / 2);
    printf("\n");
    printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_valid_filter_reci_rank, l_valid_filter_rank, l_valid_filter_tot, l3_valid_filter_tot, l1_valid_filter_tot);
    printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_valid_filter_reci_rank, r_valid_filter_rank, r_valid_filter_tot, r3_valid_filter_tot, r1_valid_filter_tot);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
           (l_valid_filter_reci_rank + r_valid_filter_reci_rank) / 2, (l_valid_filter_rank + r_valid_filter_rank) / 2, (l_valid_filter_tot + r_valid_filter_tot) / 2, (l3_valid_filter_tot + r3_valid_filter_tot) / 2, (l1_valid_filter_tot + r1_valid_filter_tot) / 2);

    validMrr = (l_valid_filter_reci_rank + r_valid_filter_reci_rank) / 2;
    validMr = (l_valid_filter_rank + r_valid_filter_rank) / 2;
    validHit10 = (l_valid_filter_tot + r_valid_filter_tot) / 2;
    validHit3 = (l3_valid_filter_tot + r3_valid_filter_tot) / 2;
    validHit1 = (l1_valid_filter_tot + r1_valid_filter_tot) / 2;

    if (type_constrain)
    {
        // type constrain
        l_valid_rank_constrain /= validTotal;
        r_valid_rank_constrain /= validTotal;
        l_valid_reci_rank_constrain /= validTotal;
        r_valid_reci_rank_constrain /= validTotal;

        l_valid_tot_constrain /= validTotal;
        l3_valid_tot_constrain /= validTotal;
        l1_valid_tot_constrain /= validTotal;

        r_valid_tot_constrain /= validTotal;
        r3_valid_tot_constrain /= validTotal;
        r1_valid_tot_constrain /= validTotal;

        // with filter
        l_valid_filter_rank_constrain /= validTotal;
        r_valid_filter_rank_constrain /= validTotal;
        l_valid_filter_reci_rank_constrain /= validTotal;
        r_valid_filter_reci_rank_constrain /= validTotal;

        l_valid_filter_tot_constrain /= validTotal;
        l3_valid_filter_tot_constrain /= validTotal;
        l1_valid_filter_tot_constrain /= validTotal;

        r_valid_filter_tot_constrain /= validTotal;
        r3_valid_filter_tot_constrain /= validTotal;
        r1_valid_filter_tot_constrain /= validTotal;

        printf("type constraint results:\n");

        printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
        printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_valid_reci_rank_constrain, l_valid_rank_constrain, l_valid_tot_constrain, l3_valid_tot_constrain, l1_valid_tot_constrain);
        printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_valid_reci_rank_constrain, r_valid_rank_constrain, r_valid_tot_constrain, r3_valid_tot_constrain, r1_valid_tot_constrain);
        printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
               (l_valid_reci_rank_constrain + r_valid_reci_rank_constrain) / 2, (l_valid_rank_constrain + r_valid_rank_constrain) / 2, (l_valid_tot_constrain + r_valid_tot_constrain) / 2, (l3_valid_tot_constrain + r3_valid_tot_constrain) / 2, (l1_valid_tot_constrain + r1_valid_tot_constrain) / 2);
        printf("\n");
        printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_valid_filter_reci_rank_constrain, l_valid_filter_rank_constrain, l_valid_filter_tot_constrain, l3_valid_filter_tot_constrain, l1_valid_filter_tot_constrain);
        printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_valid_filter_reci_rank_constrain, r_valid_filter_rank_constrain, r_valid_filter_tot_constrain, r3_valid_filter_tot_constrain, r1_valid_filter_tot_constrain);
        printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
               (l_valid_filter_reci_rank_constrain + r_valid_filter_reci_rank_constrain) / 2, (l_valid_filter_rank_constrain + r_valid_filter_rank_constrain) / 2, (l_valid_filter_tot_constrain + r_valid_filter_tot_constrain) / 2, (l3_valid_filter_tot_constrain + r3_valid_filter_tot_constrain) / 2, (l1_valid_filter_tot_constrain + r1_valid_filter_tot_constrain) / 2);

        validMrrTC = (l_valid_filter_reci_rank_constrain + r_valid_filter_reci_rank_constrain) / 2;
        validMrTC = (l_valid_filter_rank_constrain + r_valid_filter_rank_constrain) / 2;
        validHit10TC = (l_valid_filter_tot_constrain + r_valid_filter_tot_constrain) / 2;
        validHit3TC = (l3_valid_filter_tot_constrain + r3_valid_filter_tot_constrain) / 2;
        validHit1TC = (l1_valid_filter_tot_constrain + r1_valid_filter_tot_constrain) / 2;
    }
}

extern "C" void valid_relation_prediction()
/*
 * Function: test_relation_prediction
 * ----------------------------
 *   In kết quả hits@10, hits@3, hit@1, mean rank, mean reciprocal rank (có type constraint)
 *
 *
 */
{
    rel_valid_rank /= validTotal;
    rel_valid_reci_rank /= validTotal;

    rel_valid_tot /= validTotal;
    rel3_valid_tot /= validTotal;
    rel1_valid_tot /= validTotal;

    // with filter
    rel_valid_filter_rank /= validTotal;
    rel_valid_filter_reci_rank /= validTotal;

    rel_valid_filter_tot /= validTotal;
    rel3_valid_filter_tot /= validTotal;
    rel1_valid_filter_tot /= validTotal;

    printf("no type constraint results:\n");

    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
           rel_valid_reci_rank, rel_valid_rank, rel_valid_tot, rel3_valid_tot, rel1_valid_tot);
    printf("\n");
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
           rel_valid_filter_reci_rank, rel_valid_filter_rank, rel_valid_filter_tot, rel3_valid_filter_tot, rel1_valid_filter_tot);
}

extern "C" REAL getValidLinkHit10(bool type_constrain = false)
/*
 * Function: getValidLinkHit10
 * ----------------------------
 *   Trả về giá trị hits@10
 *   type_constrain: có sử dụng type constraint hay không?
 */
{
    if (type_constrain)
        printf("%f\n", validHit10TC);
    return validHit10TC;

    printf("%f\n", validHit10);
    return validHit10;
}

extern "C" REAL getValidLinkHit3(bool type_constrain = false)
/*
 * Function: getTestLinkHit3
 * ----------------------------
 *   Trả về giá trị hits@3
 *   type_constrain: có sử dụng type constraint hay không?
 */
{
    if (type_constrain)
        printf("%f\n", validHit3TC);
    return validHit3TC;

    printf("%f\n", validHit3);
    return validHit3;
}

extern "C" REAL getValidLinkHit1(bool type_constrain = false)
/*
 * Function: getTestLinkHit1
 * ----------------------------
 *   Trả về giá trị hits@1
 *   type_constrain: có sử dụng type constraint hay không?
 */
{
    if (type_constrain)
        printf("%f\n", validHit1TC);
    return validHit1TC;

    printf("%f\n", validHit1);
    return validHit1;
}

extern "C" REAL getValidLinkMR(bool type_constrain = false)
/*
 * Function: getTestLinkMR
 * ----------------------------
 *   Trả về giá trị mean rank
 *   type_constrain: có sử dụng type constraint hay không?
 */
{
    if (type_constrain)
        printf("%f\n", validMrTC);
    return validMrTC;

    printf("%f\n", validMr);
    return validMr;
}

extern "C" REAL getValidLinkMRR(bool type_constrain = false)
/*
 * Function: getTestLinkMRR
 * ----------------------------
 *   Trả về giá trị mean reciprocal rank
 *   type_constrain: có sử dụng type constraint hay không?
 */
{
    if (type_constrain)
        printf("%f\n", validMrrTC);
    return validMrrTC;

    return validMrr;
}

/*=====================================================================================
triple classification
======================================================================================*/
Triple *negValidList = NULL;

extern "C" void getNegValid()
{
    if (negValidList == NULL)
        negValidList = (Triple *)calloc(validTotal, sizeof(Triple));

    for (INT i = 0; i < validTotal; i++)
    {
        negValidList[i] = validList[i];

        if (randd(0) % 1000 < 500)
            negValidList[i].t = corrupt_head(0, validList[i].h, validList[i].r);
        else
            negValidList[i].h = corrupt_tail(0, validList[i].t, validList[i].r);
    }
}

extern "C" void getValidBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr)
{
    getNegValid();
    for (INT i = 0; i < validTotal; i++)
    {
        ph[i] = validList[i].h;
        pt[i] = validList[i].t;
        pr[i] = validList[i].r;
        
        nh[i] = negValidList[i].h;
        nt[i] = negValidList[i].t;
        nr[i] = negValidList[i].r;
    }
}

#endif