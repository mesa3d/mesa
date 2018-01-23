"""SCons.Tool.gcc

Tool-specific initialization for vali / clang

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""

#
# Copyright (c) 2001, 2002, 2003, 2004 The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import sys
import os
import os.path
import string
import subprocess

import SCons.Action
import SCons.Builder
import SCons.Tool
import SCons.Util
import SCons.Node
from SCons.Tool import createObjBuilders, createCFileBuilders

def compile_type(lang, target, source_object, env):
    code = 1
    if isinstance(source_object, str):
        objfile = ""
        if isinstance(target, SCons.Node.FS.Entry): 
            target = target.disambiguate()
            return compile_type(lang, target, source_object, env)
        elif isinstance(target, SCons.Node.FS.Dir):
            objfile = target.get_abspath() + os.path.basename(source_object)
        elif isinstance(target, SCons.Node.FS.File):
            objfile = target.get_abspath()
        elif isinstance(target, str):
            if os.path.isdir(target) or '.' not in target:
                objfile = os.path.join(target, os.path.basename(source_object))
            else:
                objfile = target
        else:
            print("unknown target-type: " + str(type(target)))
            return code

        # Create directory if doesn't exist
        if not os.path.exists(os.path.dirname(objfile)):
            os.makedirs(os.path.dirname(objfile))
        
        # Replace extension with .o
        objfile = objfile[0:objfile.rfind('.')] + ".o"

        # Build argument string
        if lang == "c":
            Compiler = env['CC'] + " -c " + env['CFLAGS'][0]
        elif lang == "cpp":
            Compiler = env['CXX'] + " -c " + env['CXXFLAGS'][0]
        else:
            print("Unknown language: " + lang)
            sys.exit(1)
        IncludePaths = ""
        CFile = ' -o ' + objfile + ' ' + source_object

        # Add all CPPPATHS
        base_path = os.getcwd()
        while not base_path.endswith('mesa'):
            base_path = os.path.normpath(os.path.join(base_path, '..'))
        for inc in env['CPPPATH']:
            try: 
                resolved_inc = inc.replace('#', '')
                if resolved_inc.startswith('/'):
                    resolved_inc.replace('/', '', 1)
                if os.path.exists(os.path.join(base_path, resolved_inc)):
                    IncludePaths += ' -I' + os.path.join(base_path, resolved_inc)
            except Exception:
                pass

        # Compile object
        print("Compiling " + lang + " source object: " + source_object)
        code = subprocess.call(Compiler + IncludePaths + CFile, shell=True)
        if code is not 0:
            return code
    elif isinstance(source_object, SCons.Node.FS.Entry):
        source_object = source_object.disambiguate(must_exist=True)
        return compile_type(lang, target, source_object, env)
    elif isinstance(source_object, SCons.Node.FS.Dir):
        for _dir in source_object.getRepositories():
            print("compile directory " + str(_dir))
    elif isinstance(source_object, SCons.Node.FS.File):
        return compile_type(lang, target, source_object.get_abspath(), env)
    else:
        print("unknown type to compile: " + str(type(source_object)))
    return code

def c_emitter(target, source, env):
    for srcfile in source:
        code = compile_type(srcfile.srcnode().abspath[srcfile.srcnode().abspath.rfind('.') + 1:], target[0], srcfile.srcnode().abspath, env)
        if code != 0:
            sys.exit(code)
    return (target, source)

def cxx_emitter(target, source, env):
    for srcfile in source:
        code = compile_type(srcfile.srcnode().abspath[srcfile.srcnode().abspath.rfind('.') + 1:], target[0], srcfile.srcnode().abspath, env)
        if code != 0:
            sys.exit(code)
    return (target, source)

def lib_emitter(target, source, env):
    # Link objects
    LinkFlags = " /lib /out:" + target[0].abspath + " "
    for obj in source:
        LinkFlags += " " + obj.abspath
    print("Linking static library: " + target[0].abspath)
    code = subprocess.call(env['LD'] + LinkFlags, shell=True)
    if code is not 0:
        sys.exit(code)
    return (target, source)

def get_link_options(env):
    ExtraLibraries = ""
    for lib in env['LIBS']:
        if isinstance(lib, SCons.Node.NodeList):
            for innerlib in lib:
                ExtraLibraries += " " + innerlib.abspath
    LinkFlags = env['LINK_FLAGS'] + ExtraLibraries
    return LinkFlags

def shlib_emitter(target, source, env):
    # Link objects
    LinkFlags = get_link_options(env) + " /dll /lldmap /entry:__CrtLibraryEntry /out:" + target[0].abspath + " "
    for obj in source:
        LinkFlags += " " + obj.abspath
    print("Linking shared library: " + target[0].abspath)
    code = subprocess.call(env['LD'] + LinkFlags, shell=True)
    if code is not 0:
        sys.exit(code)
    return (target, source)

def program_emitter(target, source, env):
    # Link objects
    LinkFlags = get_link_options(env) + " /lldmap /entry:__CrtConsoleEntry /out:" + target[0].abspath + " "
    for obj in source:
        LinkFlags += " " + obj.abspath
    print("Linking application: " + target[0].abspath)
    code = subprocess.call(env['LD'] + LinkFlags, shell=True)
    if code is not 0:
        sys.exit(code)
    return (target, source)

def generate(env):
    # Most of vali is the same as clang and friends...
    #vali_tools = ['clang', 'clang++', 'lld-link', 'ar', 'ranlib']
    vali_tools = ['gcc', 'g++', 'gnulink', 'ar', 'gas', 'lex', 'yacc']
    for tool in vali_tools:
        SCons.Tool.Tool(tool)(env)
    
    # Setup tools
    env['AS']                       = os.path.join(os.getenv("CROSS"), "bin", "clang")
    env['CC']                       = os.path.join(os.getenv("CROSS"), "bin", "clang")
    env['CXX']                      = os.path.join(os.getenv("CROSS"), "bin", "clang++")
    env['LD']                       = os.path.join(os.getenv("CROSS"), "bin", "lld-link")
    env['AR']                       = "ar"
    env['RANLIB']                   = "ranlib"

    # Setup compilation flags
    IncludePaths                    = ' -I' + os.path.join(os.getenv("INCLUDES"), "cxx") + ' -I' + os.getenv("INCLUDES")
    if str(os.getenv("VALI_ARCH")) == "i386":
        env['CFLAGS']                   = "-U_WIN32 -DMOLLENOS -DZLIB_DLL -Di386 -D__i386__ -m32 -fms-extensions -Wall -nostdlib -nostdinc -O3" + IncludePaths
        env['CXXFLAGS']                 = "-U_WIN32 -DMOLLENOS -DZLIB_DLL -Di386 -D__i386__ -m32 -fms-extensions -Wall -nostdlib -nostdinc -O3" + IncludePaths
        env['LINK_FLAGS']               = ' /nodefaultlib /machine:X86 /subsystem:native'
    elif str(os.getenv("VALI_ARCH")) == "amd64":
        env['CFLAGS']                   = "-U_WIN32 -DMOLLENOS -DZLIB_DLL -Damd64 -D__amd64__ -m64 -fms-extensions -Wall -nostdlib -nostdinc -O3" + IncludePaths
        env['CXXFLAGS']                 = "-U_WIN32 -DMOLLENOS -DZLIB_DLL -Damd64 -D__amd64__ -m64 -fms-extensions -Wall -nostdlib -nostdinc -O3" + IncludePaths
        env['LINK_FLAGS']               = ' /nodefaultlib /machine:X64 /subsystem:native'
    
    env['LINK_FLAGS']                   += " " + os.path.join(os.getenv("LIBRARIES"), "libclang.lib")
    env['LINK_FLAGS']                   += " " + os.path.join(os.getenv("LIBRARIES"), "libm.lib")
    env['LINK_FLAGS']                   += " " + os.path.join(os.getenv("LIBRARIES"), "libc.lib")
    env['LINK_FLAGS']                   += " " + os.path.join(os.getenv("LIBRARIES"), "libcrt.lib")
    env['LINK_FLAGS']                   += " " + os.path.join(os.getenv("LIBRARIES"), "zlib.lib")
    env['LINK_FLAGS']                   += " " + os.path.join(os.getenv("LIBRARIES"), "libunwind.lib")

    env['WIN32DEFPREFIX']           = ''
    env['WIN32DEFSUFFIX']           = '.def'
    env['SHOBJSUFFIX']              = '.o'
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1
    
    # Some setting from the platform also have to be overridden:
    env['OBJPREFIX']                = ''
    env['OBJSUFFIX']                = '.o'
    env['SHOBJPREFIX']              = '$OBJPREFIX'
    env['SHOBJSUFFIX']              = '$OBJSUFFIX'
    env['PROGPREFIX']               = ''
    env['PROGSUFFIX']               = '.app'
    env['LIBPREFIX']                = 'lib'
    env['LIBSUFFIX']                = '.a'
    env['SHLIBPREFIX']              = ''
    env['SHLIBSUFFIX']              = '.dll'
    env['LIBPREFIXES']              = [ 'lib', '' ]
    env['LIBSUFFIXES']              = [ '.a', '.lib' ]

    # Setup emitters
    env.Append(PROGEMITTER          = [program_emitter])
    env.Append(SHLIBEMITTER         = [shlib_emitter])
    env.Append(LIBEMITTER           = [lib_emitter])
    env.CFile                       = SCons.Builder.Builder(action = c_emitter)
    env.CXXFile                     = SCons.Builder.Builder(action = cxx_emitter)

    # Get the underlying builder objects
    static_obj, shared_obj = createObjBuilders(env)

    static_obj.add_emitter('.c', c_emitter)
    shared_obj.add_emitter('.c', c_emitter)

    static_obj.add_emitter('.cpp', cxx_emitter)
    shared_obj.add_emitter('.cpp', cxx_emitter)

def exists(env):
    return os.getenv("VALI_ARCH") is not None
