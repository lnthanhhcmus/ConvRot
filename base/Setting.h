#ifndef SETTING_H
#define SETTING_H
#define INT long
#define REAL float
#include <cstring>
#include <cstdio>
#include <string>

// Set các biến toàn cục
std::string inPath = "../data/FB15K/";
std::string outPath = "../data/FB15K/";
std::string ent_file = "";
std::string rel_file = "";
std::string train_file = "";
std::string valid_file = "";
std::string test_file = "";

extern "C" void setInPath(char *path)
/*
 * Function: setInPath
 * ----------------------------
 *   Hàm set đường dẫn input
 *   path: chuỗi chứa thông tin đường dẫn input
 */
{
	INT len = strlen(path);
	inPath = "";
	for (INT i = 0; i < len; i++)
		inPath = inPath + path[i];
	printf("Input Files Path : %s\n", inPath.c_str());
}

extern "C" void setOutPath(char *path)
/*
 * Function: setOutPath
 * ----------------------------
 *   Hàm set đường dẫn output
 *   path: chuỗi chứa thông tin đường dẫn output
 */
{
	INT len = strlen(path);
	outPath = "";
	for (INT i = 0; i < len; i++)
		outPath = outPath + path[i];
	printf("Output Files Path : %s\n", outPath.c_str());
}

extern "C" void setTrainPath(char *path)
/*
 * Function: setTrainPath
 * ----------------------------
 *   Hàm set đường dẫn đến file train
 *   path: chuỗi chứa thông tin đường dẫn đến file train
 */
{
	INT len = strlen(path);
	train_file = "";
	for (INT i = 0; i < len; i++)
		train_file = train_file + path[i];
	printf("Training Files Path : %s\n", train_file.c_str());
}

extern "C" void setValidPath(char *path)
/*
 * Function: setValidPath
 * ----------------------------
 *   Hàm set đường dẫn đến file validation
 *   path: chuỗi chứa thông tin đường dẫn đến file validation
 */
{
	INT len = strlen(path);
	valid_file = "";
	for (INT i = 0; i < len; i++)
		valid_file = valid_file + path[i];
	printf("Valid Files Path : %s\n", valid_file.c_str());
}

extern "C" void setTestPath(char *path)
/*
 * Function: setTestPath
 * ----------------------------
 *   Hàm set đường dẫn đến file test
 *   path: chuỗi chứa thông tin đường dẫn đến file test
 */
{
	INT len = strlen(path);
	test_file = "";
	for (INT i = 0; i < len; i++)
		test_file = test_file + path[i];
	printf("Test Files Path : %s\n", test_file.c_str());
}

extern "C" void setEntPath(char *path)
/*
 * Function: setEntPath
 * ----------------------------
 *   Hàm set đường dẫn đến file thông tin entity
 *   path: chuỗi chứa thông tin đường dẫn đến file thông tin entity
 */
{
	INT len = strlen(path);
	ent_file = "";
	for (INT i = 0; i < len; i++)
		ent_file = ent_file + path[i];
	printf("Entity Files Path : %s\n", ent_file.c_str());
}

extern "C" void setRelPath(char *path)
/*
 * Function: setRelPath
 * ----------------------------
 *   Hàm set đường dẫn đến file thông tin relation
 *   path: chuỗi chứa thông tin đường dẫn đến file thông tin relation
 */
{
	INT len = strlen(path);
	rel_file = "";
	for (INT i = 0; i < len; i++)
		rel_file = rel_file + path[i];
	printf("Relation Files Path : %s\n", rel_file.c_str());
}

/*
============================================================
*/

// Khai báo số lượng work threads
INT workThreads = 1;

extern "C" void setWorkThreads(INT threads)
/*
 * Function: setWorkThreads
 * ----------------------------
 *   Hàm gán số thread huấn luyện mô hình
 *   threads: số luồng sử dụng
 */
{
	workThreads = threads;
}

extern "C" INT getWorkThreads()
/*
 * Function: getWorkThreads
 * ----------------------------
 *   Trả về số luồng thực thi huấn luyện mô hình
 */
{
	return workThreads;
}

/*
============================================================
*/

// Khai báo biến toàn cục 
INT relationTotal = 0; // số lượng quan hệ 
INT entityTotal = 0; // số lượng thực thể
INT tripleTotal = 0; // số lượng bộ ba
INT testTotal = 0; // số lượng bộ ba kiểm tra
INT trainTotal = 0; // số lượng bộ ba huấn luyện
INT validTotal = 0; // số lượng bộ ba xác nhận

extern "C" INT getEntityTotal()
/*
 * Function: getEntityTotal
 * ----------------------------
 *   Trả về số lượng thực thể
 */
{
	return entityTotal;
}

extern "C" INT getRelationTotal()
/*
 * Function: getRelationTotal
 * ----------------------------
 *   Trả về số lượng quan hệ
 */
{
	return relationTotal;
}

extern "C" INT getTripleTotal()
/*
 * Function: getTripleTotal
 * ----------------------------
 *   Trả về số lượng bộ ba
 */
{
	return tripleTotal;
}

extern "C" INT getTrainTotal()
/*
 * Function: getTrainTotal
 * ----------------------------
 *   Trả về số lượng bộ ba huấn luyện
 */
{
	return trainTotal;
}

extern "C" INT getTestTotal()
/*
 * Function: getTestTotal
 * ----------------------------
 *   Trả về số lượng bộ ba kiểm tra
 */
{
	return testTotal;
}

extern "C" INT getValidTotal()
/*
 * Function: getValidTotal
 * ----------------------------
 *   Trả về số lượng bộ ba xác nhận
 */
{
	return validTotal;
}
/*
============================================================
*/

// Biến toàn cục cho lấy mẫu âm theo phân phối Bernoulli
INT bernFlag = 0;

extern "C" void setBern(INT con)
{
	bernFlag = con;
}

#endif
