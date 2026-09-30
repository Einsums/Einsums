//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
//----------------------------------------------------------------------------------------------

#ifndef EINSUMS_WINDOWS_TYPE_DEFINITIONS
#define EINSUMS_WINDOWS_TYPE_DEFINITIONS

#include <stdint.h>
#include <wchar.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef WINAPI
#    define WINAPI __stdcall
#endif

#define MAX_PATH 260

// Copy-paste of the provided Windows IDL.
// typedef unsigned short wchar_t;
typedef void         *ADCONNECTION_HANDLE;
typedef int           BOOL, *PBOOL, *LPBOOL;
typedef unsigned char BYTE, *PBYTE, *LPBYTE;
typedef BYTE          BOOLEAN, *PBOOLEAN;
typedef wchar_t       WCHAR, *PWCHAR;
typedef WCHAR        *BSTR;
typedef char          CHAR, *PCHAR;
typedef double        DOUBLE;
typedef unsigned long DWORD, *PDWORD, *LPDWORD;
typedef unsigned int  DWORD32;
typedef uint64_t      DWORD64, *PDWORD64;
typedef uint64_t      ULONGLONG;
typedef ULONGLONG     DWORDLONG, *PDWORDLONG;
typedef unsigned long error_status_t;
typedef float         FLOAT;
typedef unsigned char UCHAR, *PUCHAR;
typedef short         SHORT;

typedef void            *HANDLE;
typedef DWORD            HCALL;
typedef int              INT, *LPINT;
typedef signed char      INT8;
typedef signed short     INT16;
typedef signed int       INT32;
typedef int64_t          INT64;
typedef void            *LDAP_UDP_HANDLE;
typedef wchar_t const   *LMCSTR;
typedef WCHAR           *LMSTR;
typedef long             LONG, *PLONG, *LPLONG;
typedef signed long long LONGLONG;
typedef LONG             HRESULT;

typedef signed long long   LONG_PTR;
typedef unsigned long long ULONG_PTR;

typedef signed int       LONG32;
typedef signed long long LONG64, *PLONG64;
typedef char const      *LPCSTR;
typedef void const      *LPCVOID;
typedef wchar_t const   *LPCWSTR;
typedef char            *PSTR, *LPSTR;

typedef wchar_t         *LPWSTR, *PWSTR;
typedef DWORD            NET_API_STATUS;
typedef long             NTSTATUS;
typedef void            *PCONTEXT_HANDLE;
typedef PCONTEXT_HANDLE *PPCONTEXT_HANDLE;

typedef uint64_t QWORD;
typedef void    *RPC_BINDING_HANDLE;
typedef UCHAR   *STRING;

typedef unsigned int   UINT;
typedef unsigned char  UINT8;
typedef unsigned short UINT16;
typedef unsigned int   UINT32;
typedef uint64_t       UINT64;
typedef unsigned long  ULONG, *PULONG;

typedef ULONG_PTR      DWORD_PTR;
typedef ULONG_PTR      SIZE_T;
typedef unsigned int   ULONG32;
typedef uint64_t       ULONG64;
typedef wchar_t        UNICODE;
typedef unsigned short USHORT;
typedef void           VOID, *PVOID, *LPVOID;
typedef unsigned short WORD, *PWORD, *LPWORD;

typedef struct _FILETIME {
    DWORD dwLowDateTime;
    DWORD dwHighDateTime;
} FILETIME, *PFILETIME, *LPFILETIME;

typedef struct _GUID {
    unsigned long  Data1;
    unsigned short Data2;
    unsigned short Data3;
    BYTE           Data4[8];
} GUID, UUID, *PGUID;

typedef struct _LARGE_INTEGER {
    int64_t QuadPart;
} LARGE_INTEGER, *PLARGE_INTEGER;

typedef struct _EVENT_DESCRIPTOR {
    USHORT    Id;
    UCHAR     Version;
    UCHAR     Channel;
    UCHAR     Level;
    UCHAR     Opcode;
    USHORT    Task;
    ULONGLONG Keyword;
} EVENT_DESCRIPTOR, *PEVENT_DESCRIPTOR, *PCEVENT_DESCRIPTOR;

typedef struct _EVENT_HEADER {
    USHORT           Size;
    USHORT           HeaderType;
    USHORT           Flags;
    USHORT           EventProperty;
    ULONG            ThreadId;
    ULONG            ProcessId;
    LARGE_INTEGER    TimeStamp;
    GUID             ProviderId;
    EVENT_DESCRIPTOR EventDescriptor;
    union {
        struct {
            ULONG KernelTime;
            ULONG UserTime;
        };
        ULONG64 ProcessorTime;
    };
    GUID ActivityId;
} EVENT_HEADER, *PEVENT_HEADER;

typedef DWORD LCID;

typedef struct _LUID {
    DWORD LowPart;
    LONG  HighPart;
} LUID, *PLUID;

typedef struct _MULTI_SZ {
    wchar_t *Value;
    DWORD    nChar;
} MULTI_SZ;

typedef struct _RPC_UNICODE_STRING {
    unsigned short Length;
    unsigned short MaximumLength;
    WCHAR         *Buffer;
} RPC_UNICODE_STRING, *PRPC_UNICODE_STRING;

typedef struct _SERVER_INFO_100 {
    DWORD    sv100_platform_id;
    wchar_t *sv100_name;
} SERVER_INFO_100, *PSERVER_INFO_100, *LPSERVER_INFO_100;

typedef struct _SERVER_INFO_101 {
    DWORD    sv101_platform_id;
    wchar_t *sv101_name;
    DWORD    sv101_version_major;
    DWORD    sv101_version_minor;
    DWORD    sv101_version_type;
    wchar_t *sv101_comment;
} SERVER_INFO_101, *PSERVER_INFO_101, *LPSERVER_INFO_101;

typedef struct _SYSTEMTIME {
    WORD wYear;
    WORD wMonth;
    WORD wDayOfWeek;
    WORD wDay;
    WORD wHour;
    WORD wMinute;
    WORD wSecond;
    WORD wMilliseconds;
} SYSTEMTIME, *PSYSTEMTIME;

typedef struct _UINT128 {
    UINT64 lower;
    UINT64 upper;
} UINT128, *PUINT128;

typedef struct _ULARGE_INTEGER {
    uint64_t QuadPart;
} ULARGE_INTEGER, *PULARGE_INTEGER;

typedef struct _RPC_SID_IDENTIFIER_AUTHORITY {
    BYTE Value[6];
} RPC_SID_IDENTIFIER_AUTHORITY;

typedef DWORD        ACCESS_MASK;
typedef ACCESS_MASK *PACCESS_MASK;

typedef struct _OBJECT_TYPE_LIST {
    WORD        Level;
    ACCESS_MASK Remaining;
    GUID       *ObjectType;
} OBJECT_TYPE_LIST, *POBJECT_TYPE_LIST;

typedef struct _ACE_HEADER {
    UCHAR  AceType;
    UCHAR  AceFlags;
    USHORT AceSize;
} ACE_HEADER, *PACE_HEADER;

typedef struct _SYSTEM_MANDATORY_LABEL_ACE {
    ACE_HEADER  Header;
    ACCESS_MASK Mask;
    DWORD       SidStart;
} SYSTEM_MANDATORY_LABEL_ACE, *PSYSTEM_MANDATORY_LABEL_ACE;

typedef struct _TOKEN_MANDATORY_POLICY {
    DWORD Policy;
} TOKEN_MANDATORY_POLICY, *PTOKEN_MANDATORY_POLICY;

typedef struct _MANDATORY_INFORMATION {
    ACCESS_MASK            AllowedAccess;
    BOOLEAN                WriteAllowed;
    BOOLEAN                ReadAllowed;
    BOOLEAN                ExecuteAllowed;
    TOKEN_MANDATORY_POLICY MandatoryPolicy;
} MANDATORY_INFORMATION, *PMANDATORY_INFORMATION;

typedef struct _CLAIM_SECURITY_ATTRIBUTE_OCTET_STRING_RELATIVE {
    DWORD Length;
    BYTE  OctetString[];
} CLAIM_SECURITY_ATTRIBUTE_OCTET_STRING_RELATIVE, *PCLAIM_SECURITY_ATTRIBUTE_OCTET_STRING_RELATIVE;

typedef struct _CLAIM_SECURITY_ATTRIBUTE_RELATIVE_V1 {
    DWORD Name;
    WORD  ValueType;
    WORD  Reserved;
    DWORD Flags;
    DWORD ValueCount;
    union {
        PLONG64                                         pInt64[];
        PDWORD64                                        pUint64[];
        PWSTR                                           ppString[];
        PCLAIM_SECURITY_ATTRIBUTE_OCTET_STRING_RELATIVE pOctetString[];
    } Values;
} CLAIM_SECURITY_ATTRIBUTE_RELATIVE_V1, *PCLAIM_SECURITY_ATTRIBUTE_RELATIVE_V1;

typedef DWORD SECURITY_INFORMATION, *PSECURITY_INFORMATION;

typedef struct _RPC_SID {
    unsigned char                Revision;
    unsigned char                SubAuthorityCount;
    RPC_SID_IDENTIFIER_AUTHORITY IdentifierAuthority;
    unsigned long                SubAuthority[];
} RPC_SID, *PRPC_SID, *PSID;

typedef struct _ACL {
    unsigned char  AclRevision;
    unsigned char  Sbz1;
    unsigned short AclSize;
    unsigned short AceCount;
    unsigned short Sbz2;
} ACL, *PACL;

typedef struct _SECURITY_DESCRIPTOR {
    UCHAR  Revision;
    UCHAR  Sbz1;
    USHORT Control;
    PSID   Owner;
    PSID   Group;
    PACL   Sacl;
    PACL   Dacl;
} SECURITY_DESCRIPTOR, *PSECURITY_DESCRIPTOR;

typedef HANDLE HINSTANCE;

typedef HINSTANCE HMODULE;

#define INVALID_HANDLE_VALUE ((HANDLE)(LONG_PTR)(-1))

#ifdef __cplusplus
}
#endif
#endif
