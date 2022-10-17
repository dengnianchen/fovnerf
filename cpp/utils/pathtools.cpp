//========= Copyright Valve Corporation ============//
#include "compat.h"
#include "strtools.h"
#include "pathtools.h"

#if defined( _WIN32)
#include <windows.h>
#include <direct.h>
#include <shobjidl.h>
#include <knownfolders.h>
#include <shlobj.h>
#include <share.h>

#undef GetEnvironmentVariable
#else
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#endif
#if defined OSX
#include <Foundation/Foundation.h>
#include <AppKit/AppKit.h>
#include <mach-o/dyld.h>
#define _S_IFDIR S_IFDIR     // really from tier0/platform.h which we dont have yet
#endif

#include <sys/stat.h>

#include <algorithm>


/** Returns the path of the current working directory */
std::string Path_GetWorkingDirectory()
{
	std::string sPath;
#if defined( _WIN32 )
	wchar_t buf[MAX_UNICODE_PATH];
	sPath = UTF16to8( _wgetcwd( buf, MAX_UNICODE_PATH ) );
#else
	char buf[ 1024 ];
	sPath = getcwd( buf, sizeof( buf ) );
#endif
	return sPath;
}

/** Sets the path of the current working directory. Returns true if this was successful. */
bool Path_SetWorkingDirectory( const std::string & sPath )
{
	bool bSuccess;
#if defined( _WIN32 )
	std::wstring wsPath = UTF8to16( sPath.c_str() );
	bSuccess = 0 == _wchdir( wsPath.c_str() );
#else
	bSuccess = 0 == chdir( sPath.c_str() );
#endif
	return bSuccess;
}

/** Returns the specified path without its filename */
std::string Path_StripFilename( const std::string & sPath, char slash )
{
	if( slash == 0 )
		slash = Path_GetSlash();

	std::string::size_type n = sPath.find_last_of( slash );
	if( n == std::string::npos )
		return sPath;
	else
		return std::string( sPath.begin(), sPath.begin() + n );
}

/** returns just the filename from the provided full or relative path. */
std::string Path_StripDirectory( const std::string & sPath, char slash )
{
	if( slash == 0 )
		slash = Path_GetSlash();

	std::string::size_type n = sPath.find_last_of( slash );
	if( n == std::string::npos )
		return sPath;
	else
		return std::string( sPath.begin() + n + 1, sPath.end() );
}

/** returns just the filename with no extension of the provided filename. 
* If there is a path the path is left intact. */
std::string Path_StripExtension( const std::string & sPath )
{
	for( std::string::const_reverse_iterator i = sPath.rbegin(); i != sPath.rend(); i++ )
	{
		if( *i == '.' )
		{
			return std::string( sPath.begin(), i.base() - 1 );
		}

		// if we find a slash there is no extension
		if( *i == '\\' || *i == '/' )
			break;
	}

	// we didn't find an extension
	return sPath;
}

/** returns just extension of the provided filename (if any). */
std::string Path_GetExtension( const std::string & sPath )
{
	for ( std::string::const_reverse_iterator i = sPath.rbegin(); i != sPath.rend(); i++ )
	{
		if ( *i == '.' )
		{
			return std::string( i.base(), sPath.end() );
		}

		// if we find a slash there is no extension
		if ( *i == '\\' || *i == '/' )
			break;
	}

	// we didn't find an extension
	return "";
}

bool Path_IsAbsolute( const std::string & sPath )
{
	if( sPath.empty() )
		return false;

#if defined( WIN32 )
	if ( sPath.size() < 3 ) // must be c:\x or \\x at least
		return false;

	if ( sPath[1] == ':' ) // drive letter plus slash, but must test both slash cases
	{
		if ( sPath[2] == '\\' || sPath[2] == '/' )
			return true;
	}
	else if ( sPath[0] == '\\' && sPath[1] == '\\' ) // UNC path
		return true;
#else
	if( sPath[0] == '\\' || sPath[0] == '/' ) // any leading slash
		return true;
#endif

	return false;
}


/** Makes an absolute path from a relative path and a base path */
std::string Path_MakeAbsolute( const std::string & sRelativePath, const std::string & sBasePath )
{
	if( Path_IsAbsolute( sRelativePath ) )
		return sRelativePath;
	else
	{
		if( !Path_IsAbsolute( sBasePath ) )
			return "";

		std::string sCompacted = Path_Compact( Path_Join( sBasePath, sRelativePath ) );
		if( Path_IsAbsolute( sCompacted ) )
			return sCompacted;
		else
			return "";
	}
}


/** Fixes the directory separators for the current platform */
std::string Path_FixSlashes( const std::string & sPath, char slash )
{
	if( slash == 0 )
		slash = Path_GetSlash();

	std::string sFixed = sPath;
	for( std::string::iterator i = sFixed.begin(); i != sFixed.end(); i++ )
	{
		if( *i == '/' || *i == '\\' )
			*i = slash;
	}

	return sFixed;
}


char Path_GetSlash()
{
#if defined(_WIN32)
	return '\\';
#else
	return '/';
#endif
}

/** Jams two paths together with the right kind of slash */
std::string Path_Join( const std::string & first, const std::string & second, char slash )
{
	if( slash == 0 )
		slash = Path_GetSlash();

	// only insert a slash if we don't already have one
	std::string::size_type nLen = first.length();
	if( !nLen )
		return second;
#if defined(_WIN32)
	if( first.back() == '\\' || first.back() == '/' )
	    nLen--;
#else
	char last_char = first[first.length()-1];
	if (last_char == '\\' || last_char == '/')
	    nLen--;
#endif

	return first.substr( 0, nLen ) + std::string( 1, slash ) + second;
}


std::string Path_Join( const std::string & first, const std::string & second, const std::string & third, char slash )
{
	return Path_Join( Path_Join( first, second, slash ), third, slash );
}

std::string Path_Join( const std::string & first, const std::string & second, const std::string & third, const std::string &fourth, char slash )
{
	return Path_Join( Path_Join( Path_Join( first, second, slash ), third, slash ), fourth, slash );
}

std::string Path_Join( 
	const std::string & first, 
	const std::string & second, 
	const std::string & third, 
	const std::string & fourth, 
	const std::string & fifth, 
	char slash )
{
	return Path_Join( Path_Join( Path_Join( Path_Join( first, second, slash ), third, slash ), fourth, slash ), fifth, slash );
}


std::string Path_RemoveTrailingSlash( const std::string & sRawPath, char slash )
{
	if ( slash == 0 )
		slash = Path_GetSlash();

	std::string sPath = sRawPath;
	std::string::size_type nCurrent = sRawPath.length();
	if ( nCurrent == 0 )
		return sPath;

	int nLastFound = -1;
	nCurrent--;
	while( nCurrent != 0 )
	{
		if ( sRawPath[ nCurrent ] == slash )
		{
			nLastFound = (int)nCurrent;
			nCurrent--;
		}
		else
		{
			break;
		}
	}
		
	if ( nLastFound >= 0 )
	{
		sPath.erase( nLastFound, std::string::npos );
	}
	
	return sPath;
}


/** Removes redundant <dir>/.. elements in the path. Returns an empty path if the 
* specified path has a broken number of directories for its number of ..s */
std::string Path_Compact( const std::string & sRawPath, char slash )
{
	if( slash == 0 )
		slash = Path_GetSlash();

	std::string sPath = Path_FixSlashes( sRawPath, slash );
	std::string sSlashString( 1, slash );

	// strip out all /./
	for( std::string::size_type i = 0; (i + 3) < sPath.length();  )
	{
		if( sPath[ i ] == slash && sPath[ i+1 ] == '.' && sPath[ i+2 ] == slash )
		{
			sPath.replace( i, 3, sSlashString );
		}
		else
		{
			++i;
		}
	}


	// get rid of trailing /. but leave the path separator
	if( sPath.length() > 2 )
	{
		std::string::size_type len = sPath.length();
		if( sPath[ len-1 ] == '.'  && sPath[ len-2 ] == slash )
		{
		  // sPath.pop_back();
		  sPath[len-1] = 0;  // for now, at least
		}
	}

	// get rid of leading ./ 
	if( sPath.length() > 2 )
	{
		if( sPath[ 0 ] == '.'  && sPath[ 1 ] == slash )
		{
			sPath.replace( 0, 2, "" );
		}
	}

	// each time we encounter .. back up until we've found the previous directory name
	// then get rid of both
	std::string::size_type i = 0;
	while( i < sPath.length() )
	{
		if( i > 0 && sPath.length() - i >= 2 
			&& sPath[i] == '.'
			&& sPath[i+1] == '.'
			&& ( i + 2 == sPath.length() || sPath[ i+2 ] == slash )
			&& sPath[ i-1 ] == slash )
		{
			// check if we've hit the start of the string and have a bogus path
			if( i == 1 )
				return "";
			
			// find the separator before i-1
			std::string::size_type iDirStart = i-2;
			while( iDirStart > 0 && sPath[ iDirStart - 1 ] != slash )
				--iDirStart;

			// remove everything from iDirStart to i+2
			sPath.replace( iDirStart, (i - iDirStart) + 3, "" );

			// start over
			i = 0;
		}
		else
		{
			++i;
		}
	}

	return sPath;
}


/** Returns the path to the current DLL or exe */
std::string Path_GetThisModulePath()
{
	// gets the path of vrclient.dll itself
#ifdef WIN32
	HMODULE hmodule = NULL;

	::GetModuleHandleEx( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCTSTR>(Path_GetThisModulePath), &hmodule );

	wchar_t *pwchPath = new wchar_t[MAX_UNICODE_PATH];
	char *pchPath = new char[ MAX_UNICODE_PATH_IN_UTF8 ];
	::GetModuleFileNameW( hmodule, pwchPath, MAX_UNICODE_PATH );
	WideCharToMultiByte( CP_UTF8, 0, pwchPath, -1, pchPath, MAX_UNICODE_PATH_IN_UTF8, NULL, NULL );
	delete[] pwchPath;

	std::string sPath = pchPath;
	delete [] pchPath;
	return sPath;

#elif defined( OSX ) || defined( LINUX )
	// get the addr of a function in vrclient.so and then ask the dlopen system about it
	Dl_info info;
	dladdr( (void *)Path_GetThisModulePath, &info );
	return info.dli_fname;
#endif

}

/** returns true if the specified path represents an app bundle */
bool Path_IsAppBundle( const std::string & sPath )
{
#if defined(OSX)
	NSBundle *bundle = [ NSBundle bundleWithPath: [ NSString stringWithUTF8String:sPath.c_str() ] ];
	bool bisAppBundle = ( nullptr != bundle );
	[ bundle release ];
	return bisAppBundle;
#else
	return false;
#endif
}

//-----------------------------------------------------------------------------
// Purpose: returns true if the the path exists
//-----------------------------------------------------------------------------
bool Path_Exists( const std::string & sPath )
{
	std::string sFixedPath = Path_FixSlashes( sPath );
	if( sFixedPath.empty() )
		return false;

#if defined( WIN32 )
	struct	_stat	buf;
	std::wstring wsFixedPath = UTF8to16( sFixedPath.c_str() );
	if ( _wstat( wsFixedPath.c_str(), &buf ) == -1 )
	{
		return false;
	}
#else
	struct stat buf;
	if ( stat ( sFixedPath.c_str(), &buf ) == -1)
	{
		return false;
	}
#endif

	return true;
}


//-----------------------------------------------------------------------------
// Purpose: helper to find a directory upstream from a given path
//-----------------------------------------------------------------------------
std::string Path_FindParentDirectoryRecursively( const std::string &strStartDirectory, const std::string &strDirectoryName )
{
	std::string strFoundPath = "";
	std::string strCurrentPath = Path_FixSlashes( strStartDirectory );
	if ( strCurrentPath.length() == 0 )
		return "";

	bool bExists = Path_Exists( strCurrentPath );
	std::string strCurrentDirectoryName = Path_StripDirectory( strCurrentPath );
	if ( bExists && _stricmp( strCurrentDirectoryName.c_str(), strDirectoryName.c_str() ) == 0 )
		return strCurrentPath;

	while( bExists && strCurrentPath.length() != 0 )
	{
		strCurrentPath = Path_StripFilename( strCurrentPath );
		strCurrentDirectoryName = Path_StripDirectory( strCurrentPath );
		bExists = Path_Exists( strCurrentPath );
		if ( bExists && _stricmp( strCurrentDirectoryName.c_str(), strDirectoryName.c_str() ) == 0 )
			return strCurrentPath;
	}

	return "";
}


//-----------------------------------------------------------------------------
// Purpose: helper to find a subdirectory upstream from a given path
//-----------------------------------------------------------------------------
std::string Path_FindParentSubDirectoryRecursively( const std::string &strStartDirectory, const std::string &strDirectoryName )
{
	std::string strFoundPath = "";
	std::string strCurrentPath = Path_FixSlashes( strStartDirectory );
	if ( strCurrentPath.length() == 0 )
		return "";

	bool bExists = Path_Exists( strCurrentPath );
	while( bExists && strCurrentPath.length() != 0 )
	{
		strCurrentPath = Path_StripFilename( strCurrentPath );
		bExists = Path_Exists( strCurrentPath );

		if( Path_Exists( Path_Join( strCurrentPath, strDirectoryName ) ) )
		{
			strFoundPath = Path_Join( strCurrentPath, strDirectoryName );
			break;
		}
	}
	return strFoundPath;
}


#if defined(WIN32)
#define FILE_URL_PREFIX "file:///"
#else
#define FILE_URL_PREFIX "file://"
#endif

// ----------------------------------------------------------------------------------------------------------------------------
// Purpose: Turns a path to a file on disk into a URL (or just returns the value if it's already a URL)
// ----------------------------------------------------------------------------------------------------------------------------
std::string Path_FilePathToUrl( const std::string & sRelativePath, const std::string & sBasePath )
{
	if ( !strncasecmp( sRelativePath.c_str(), "http://", 7 )
		|| !strncasecmp( sRelativePath.c_str(), "https://", 8 )
		|| !strncasecmp( sRelativePath.c_str(), "file://", 7 ) )
	{
		return sRelativePath;
	}
	else
	{
		std::string sAbsolute = Path_MakeAbsolute( sRelativePath, sBasePath );
		if ( sAbsolute.empty() )
			return sAbsolute;
		sAbsolute = Path_FixSlashes( sAbsolute, '/' );
		return std::string( FILE_URL_PREFIX ) + sAbsolute;
	}
}

// -----------------------------------------------------------------------------------------------------
// Purpose: Strips off file:// off a URL and returns the path. For other kinds of URLs an empty string is returned
// -----------------------------------------------------------------------------------------------------
std::string Path_UrlToFilePath( const std::string & sFileUrl )
{
	if ( !strncasecmp( sFileUrl.c_str(), FILE_URL_PREFIX, strlen( FILE_URL_PREFIX ) ) )
	{
		std::string sRet = sFileUrl.c_str() + strlen( FILE_URL_PREFIX );
		sRet = Path_FixSlashes( sRet );
		return sRet;
	}
	else
	{
		return "";
	}
}


// -----------------------------------------------------------------------------------------------------
// Purpose: Returns the root of the directory the system wants us to store user documents in
// -----------------------------------------------------------------------------------------------------
std::string GetUserDocumentsPath()
{
#if defined( WIN32 )
	WCHAR rwchPath[MAX_PATH];

	if ( !SUCCEEDED( SHGetFolderPathW( NULL, CSIDL_MYDOCUMENTS | CSIDL_FLAG_CREATE, NULL, 0, rwchPath ) ) )
	{
		return "";
	}

	// Convert the path to UTF-8 and store in the output
	std::string sUserPath = UTF16to8( rwchPath );

	return sUserPath;
#elif defined( OSX )
	@autoreleasepool {
		NSArray *paths = NSSearchPathForDirectoriesInDomains( NSDocumentDirectory, NSUserDomainMask, YES );
		if ( [paths count] == 0 )
		{
			return "";
		}
		
		return [[paths objectAtIndex:0] UTF8String];
	}
#elif defined( LINUX )
	// @todo: not solved/changed as part of OSX - still not real - just removed old class based steam cut and paste
	const char *pchHome = getenv( "HOME" );
	if ( pchHome == NULL )
	{
		return "";
	}
	return pchHome;
#endif
}

