# Lab 2: I/O

In this lab, we implement a tool that lists all files in a directory and all its subdirectories.

You will learn
   * how to iterate through all files in a directory
   * how to retrieve the metadata of a file
   * how to print nicely formatted output
   * that error handling requires a significant effort
   * that string handling is not one of C's strenghs
   * and a bunch of other useful programming tricks and C library functions

## Important Dates

| Date | Description |
|:---  |:--- |
| Thursday, April 10, 21:00 | Submission deadline |

## Logistics

### Hand-out

Please read this file (README.md) carefully. Additionally, the PPT file presented during the lab session is included in the distribution archive for students. However, please remember that the PPT file is based on the README.md file, and it is important to understand the contents of the README.md file.

The assignment starts by extracting the provided assign2.tar.gz file. It is recommended that you link this folder to your GitHub repository (for better readability of the README file and keep your code safe).

### Submission

As in lab-1-decomment, create a directory named after your student number, compress it up, and submit it. And, be sure to follow the following directory structure:

```
202500000_assign2
  |-dirtree.c (source file)
  `-readme
```

### Implementing the Lab on an M1/M2 Mac or custom Unix/Linux

While it is possible to implement the lab on an ARM-based Mac, MacOS is not 100% compatible with Linux and thus requires a number of changes to the code (extra header files, etc.) We recommend installing an ARM-based Linux virtual machine such as [Ubuntu Server for ARM](https://ubuntu.com/download/server/arm). On custom Intel-compatible Linux systems, you should be able to implement the lab without any modifications.

Note that we evaluate your submissions in the provided Bacchus VMs. If you not use the Bacchus VMs to develop your code, **we strongly recommend to test your submission in the Bacchus VMs** to make sure it runs correctly.

## Dirtree Overview

Our tool is called _dirtree_. Dirtree recursively traverses a directory tree and prints out a sorted list of all files.

```
$ dirtree demo
demo
  subdir1
    sparsefile
    thisisanextremelylongfilenameforsuchasimplisticfile
 subdir2
    brokenlink
    symboliclink
  subdir3
    pipe
    socket
  one
  two

```

Dirtree can also print a fancy directory tree and show details...
```
$ dirtree -t -v demo
demo
|-subdir1                                                  devel:devel           4096         8  d
| |-sparsefile                                             devel:devel           8192         8   
| `-thisisanextremelylongfilenameforsuchasimplistic...     devel:devel           1000         8   
|-subdir2                                                  devel:devel           4096         8  d
| |-brokenlink                                             devel:devel              8         0  l
| `-symboliclink                                           devel:devel              6         0  l
|-subdir3                                                  devel:devel           4096         8  d
| |-pipe                                                   devel:devel              0         0  f
| `-socket                                                 devel:devel              0         0  s
|-one                                                      devel:devel              1         8   
`-two                                                      devel:devel              2         8   
```

...or a summary of a directory:
```
$ dirtree -v -s demo
Name                                                        User:Group           Size    Blocks Type 
----------------------------------------------------------------------------------------------------
demo
  subdir1                                                  devel:devel           4096         8  d
    sparsefile                                             devel:devel           8192         8   
    thisisanextremelylongfilenameforsuchasimplistic...     devel:devel           1000         8   
  subdir2                                                  devel:devel           4096         8  d
    brokenlink                                             devel:devel              8         0  l
    symboliclink                                           devel:devel              6         0  l
  subdir3                                                  devel:devel           4096         8  d
    pipe                                                   devel:devel              0         0  f
    socket                                                 devel:devel              0         0  s
  one                                                      devel:devel              1         8   
  two                                                      devel:devel              2         8   
----------------------------------------------------------------------------------------------------
4 files, 3 directories, 2 links, 1 pipe, and 1 socket                           21497        56

```

Last but not least, dirtree can generate aggregate totals over several directories:
```
$ dirtree -v -s demo/subdir1 demo/subdir2
Name                                                        User:Group           Size    Blocks Type 
----------------------------------------------------------------------------------------------------
demo/subdir1
  sparsefile                                               devel:devel           8192         8   
  thisisanextremelylongfilenameforsuchasimplisticfile      devel:devel           1000         8   
----------------------------------------------------------------------------------------------------
2 files, 0 directories, 0 links, 0 pipes, and 0 sockets                          9192        16

Name                                                        User:Group           Size    Blocks Type 
----------------------------------------------------------------------------------------------------
demo/subdir2
  brokenlink                                               devel:devel              8         0  l
  symboliclink                                             devel:devel              6         0  l
----------------------------------------------------------------------------------------------------
0 files, 0 directories, 2 links, 0 pipes, and 0 sockets                            14         0

Analyzed 2 directories:
  total # of files:                       2
  total # of directories:                 0
  total # of links:                       2
  total # of pipes:                       0
  total # of sockets:                    0
  total file size:                     9206
  total # of blocks:                     16
```

## Dirtree Specification 

### Command line arguments

Dirtree accepts the following command line arguments
```
dirtree [Options] [Directories]
```

| Option      | Description |
|:----        |:----        |
| -h          | Help screen |
| -t          | Turn on fancy tree view |
| -v          | Turn on detailed mode |
| -s          | Turn on summary mode |

`Directories` is a list of directories that are to be traversed. Dirtree accepts up to 64 directories.
If no directory is given, then the current directory is traversed. 

### Operation

1. Dirtree traverses each directory in the list `Directories` recursively. 
2. In each directory, it enumerates all directory entries and prints them in alphabetical order. Directories are listed before files. The special entries '.' and '..' are ignored.
3. A summary is printed after each directory. If several directories are traversed, an aggregate total is printed at the end.

### Output

#### Simple mode vs. fancy tree view mode
As dirtree traverses the directory tree, it prints the names of the sorted entities in a directory. The names are indented according to the level of the subdirectory. For each additional level, the names are shifted right two positions to allow for easy visual identification of the directory structure.

**Simple mode** indents names by two spaces for each additional level.
**Fancy tree mode** prints a tree structure using only ASCII characters.

<table>
  <thead>
    <tr>
      <th align="left">Simple mode</th>
      <th align="left">Fancy tree view mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <pre>dir             <br>  subdir1<br>    subdir2<br>      file1<br>      file2<br>      file3<br>  file4</pre>
      </td>
      <td>
        <pre>dir             <br>|-subdir1<br>| `-subdir2<br>|   |-file1<br>|   |-file2<br>|   `-file3<br>`-file4</pre>
      </td>
    </tr>
  </tbody>
</table>

#### Detailed mode
In detailed mode, dirtree prints out the following additional details for each entry:
* User and group  
  Each file in Unix belongs to a user and a group. Detailed mode prints the names of the user and the group separated by a colon (:).
* Size  
  The size of the file in bytes.
* Disk blocks  
  The number of blocks this file occupies on the disk. 
* File type  
  Indicates the type of file by a single character

  | Type | Character |
  |:---  |:---------:|
  | File | _(empty)_ |
  | Directory | d |
  | Link | l |
  | Character device | c |
  | Block device | b |
  | Fifo | f |
  | Socket | s |

#### Summary mode
In summary mode, dirtree prints a header and footer around each directory and a one-liner containing statistics about the directory.

If there are more than one directories provided on the command line, an aggregate total of all listed directories is shown.
```
$ dirtree -s demo/subdir1 demo/subdir3

Name
----------------------------------------------------------------------------------------------------
demo/subdir1
  subdir2
    file1
    link
----------------------------------------------------------------------------------------------------
1 file, 1 directory, 1 link, 0 pipes, and 0 sockets

Name
----------------------------------------------------------------------------------------------------
demo/subdir3
  file2
----------------------------------------------------------------------------------------------------
1 file, 0 directories, 0 links, 0 pipes, and 0 sockets

Analyzed 2 directories:
  total # of files:                       2
  total # of directories:                 1
  total # of links:                       1
  total # of pipes:                       0
  total # of sockets:                     0
```

#### Output formatting
The output in simple mode prints all elements with the correct indentation. 
In detailed mode, the output is nicely formatted and filenames that are too long are cut and end with three dots (...). 
Unless explicitly specified, you can decide for yourself whether and how you are formatting exceptional cases (error messages, etc.)

```
$ dirtree -v -s demo2
Name                                                        User:Group           Size    Blocks Type 
----------------------------------------------------------------------------------------------------
demo2
  subdir1                                                  devel:devel           4096         8  d
    subdir2                                                devel:devel           4096         8  d
      fifo                                                 devel:devel              0         0  f
      link                                                 devel:devel             11         0  l
      socket                                               devel:devel              0         0  s
      unreasonablyextremelylongfilenamethatdoesntfi...     devel:devel              0         0
  file4                                                    devel:devel              0         0
----------------------------------------------------------------------------------------------------
2 files, 2 directories, 1 link, 1 pipe, and 1 socket                             8203        16
```

The output in detailed mode consists of the following elements (Total line length: 100):

| Output element | Width | Alignment | Action on overflow |
|:---            |:-----:|:---------:|:---      |
| Path and name  | 54    | left      | cut and end with three dots |
| User name      |  8    | right     | ignore |
| Group name     |  8    | left      | ignore |
| File size      | 10    | right     | ignore |
| Disk blocks    |  8    | right     | ignore |
| Type           |  1    |           |        |
| Summary line   | 68    | left      | cut and end with three dots |
| Total size     | 14    | right     | ignore |
| Total blocks   |  9    | right     | ignore |

The following rendering shows the output formatting in detail for each of the different elements. The two rows on top indicate the character position on a line. 
```
         1         2         3         4         5         6         7         8         9        10
1........0.........0.........0.........0.........0.........0.........0.........0.........0.........0

Name                                                        User:Group           Size    Blocks Type
----------------------------------------------------------------------------------------------------
<path and name                                       >  <  user>:<group >  <    size>  <blocks>  t
<path and name                                       >  <  user>:<group >  <    size>  <blocks>  t
...
----------------------------------------------------------------------------------------------------
<summary                                                           >   <  total size> <totblks>

```

##### Summary line
dirtree takes great care to output grammatically correct English. Zero or >=2 elements are output in plural form, while for exactly one element the singular form is used.
Compare the two summary lines:
```
0 files, 2 directories, 1 link, 1 pipe, and 1 socket

1 file, 1 directory, 2 links, 0 pipes, and 5 sockets
```

### Error handling

Errors that occur when processing a directory (permission errors) are reported in place of the entries of that directory. You can change the permission of specific directory by 'chmod 000 <dir_name>' command:
```
$ dirtree -v ~/anyone_can_access_dir/
/home/ta_sj/anyone_can_access_dir/
  only_root_can_access_dir                                 ta_sj:ta_sj           4096         8  d
    ERROR: Permission denied
  anyone_can_access_file                                   ta_sj:ta_sj              0         0
```

If an error occurs when trying to retrieve an non-existent file or directory:
```
$ dirtree -v non-existent_file
non-existent_file
  ERROR: No such file or directory
```

For any other errors, you can choose what to do (That is, other errors do not need to be handled/considered). The reference implementation aborts on most errors:
```
$ dirtree -s -v demo
Name                                                        User:Group           Size    Blocks Type 
----------------------------------------------------------------------------------------------------
demo
  subdir1                                                  devel:devel           4096         8  d
    sparsefile                                             devel:devel           8192         8   
Out of memory.
```

## Handout Overview

The handout contains the following files and directories

| File/Directory | Description |
|:---  |:--- |
| README.md | this file | 
| Makefile | Makefile driver program |
| src/dirtree.c | Skeleton for dirtree.c. Implement your solution by editing this file. |
| doc/ | Doxygen instructions, configuration file, and auto-generated documentation |
| reference/ | Reference implementation |
| tools/ | Tools to generate directory trees for testing |

### Reference implementation

The directory `reference` contains our reference implementation. You can use it to compare your output to ours.

### Tools

The `tools` directory contains tools to generate test directory trees to test your solution.

| File/Directory | Description |
|:---  |:--- |
| gentree.sh | Driver script to generate a test directory tree. |
| mksock     | Helper program to generate a Unix socket. |
| *.tree     | Script files describing the directory tree layout. |

Invoke `gentree.sh` with a script file to generate one of the provided test directory trees. 

Assuming you are located in the root directory of your I/O lab repository, use the follwing command to generate the `demo` directory tree
```bash
$ ls
dirtree.c  doc  Makefile  README.md  reference  tools
$ cd tools/
$ gentree.sh demo.tree 
Generating tree from 'tools/demo.tree'...
Done. Generated 4 files, 2 links, 1 fifos, and 1 sockets. 0 errors reported.
```

You can list the contents of the tree with the reference implementation:
```bash
$ reference/dirtree -t -v -s demo/
Name                                                        User:Group           Size    Blocks Type 
----------------------------------------------------------------------------------------------------
demo/
|-subdir1                                                  devel:devel           4096         8  d
| |-sparsefile                                             devel:devel           8192         8   
| `-thisisanextremelylongfilenameforsuchasimplistic...     devel:devel           1000         8   
|-subdir2                                                  devel:devel           4096         8  d
| |-brokenlink                                             devel:devel              8         0  l
| `-symboliclink                                           devel:devel              6         0  l
|-subdir3                                                  devel:devel           4096         8  d
| |-pipe                                                   devel:devel              0         0  f
| `-socket                                                 devel:devel              0         0  s
|-one                                                      devel:devel              1         8   
`-two                                                      devel:devel              2         8   
----------------------------------------------------------------------------------------------------
4 files, 3 directories, 2 links, 1 pipe, and 1 socket                           21497        56
```

## Your Task

Your task is to implement dirtree according to the specification above.

### Design

**Recommendation**: do not look at the provided code in `dirtree.c` yet! Think about the logical steps yourself. 
The design is the most difficult and important phase in any project - and also the phase that requires the most practice and sets apart hobby programmers from experts.

### Implementation

Once you have designed the outline of your implementation, you can start implementing it. We provide a skeleton file to help you get started.

The skeleton provides data structures to manage the statistics of a directory, a function to read the next entry from a directory while ignoring the '.' and '..' entries, a comparator function to sort the entries of a directory using quicksort, and full argument parsing and syntax helpers. 

You have to implement the following two parts:
1. in `main()`  
   Iterate through the list of directories stored in `directories`. For each directory, call `processDir()` with the appropriate parameters. Also, depending on the output mode, print header, footer, and statistics.
2. in `processDir()`  
   Open, enumerate, sort, and close the directory. Print elements one by one. Update statistics. If the element is a directory, call `processDir()` recursively.

### Hints

#### Skeleton code
The skeleton code is meant to help you get started. You can modify it in any way you see fit - or implement this lab completely from scratch.

#### C library calls

Knowing which library functions exist and how to use them is difficult at the beginning in every programming language. To help you get started, we provide a list of C library calls / system calls grouped by topic that you may find helpful to solve this lab. Read the man pages carefully to learn how exactly the functions operate.

<table>
  <thead>
    <tr>
      <th align="left">Topic</th>
      <th align="left">C library call</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">
        String operations
      </td>
      <td>
        <code>strcmp()</code>
      </td>
      <td>
        compare two strings
      </td>
    </tr>
    <tr>
      <td>
        <code>strncpy()</code>
      </td>
      <td>
        copy up to n characters of one string into another
      </td>
    </tr>
    <tr>
      <td>
        <code>strdup()</code>
      </td>
      <td>
        create a copy of a string. Use <code>free()</code> to free it after use
      </td>
    </tr>
    <tr>
      <td>
        <code>asprintf()</code>
      </td>
      <td>
        asprintf() is extremely helpful to print into a string and allocate memory for it at the same time.
        We will show some examples during the lab session.
      </td>
    </tr>
    <tr>
      <td rowspan="3">
        Directory management
      </td>
      <td>
        <code>opendir()</code>
      </td>
      <td>
        open a directory to enumerate its entries
      </td>
    </tr>
    <tr>
      <td>
        <code>closedir()</code>
      </td>
      <td>
        close an open directory
      </td>
    </tr>
    <tr>
      <td>
        <code>readdir()</code>
      </td>
      <td>
        read next entry from directory
      </td>
    </tr>
    <tr>
      <td rowspan="2">
        File meta data
      </td>
      <td>
        <code>stat()</code>
      </td>
      <td>
        retrieve meta data of a file, follow links
      </td>
    </tr>
    <tr>
      <td>
        <code>lstat()</code>
      </td>
      <td>
        retrieve meta data of a file, do not follow links
      </td>
    </tr>
    <tr>
      <td rowspan="2">
        User/group information
      </td>
      <td>
        <code>getpwuid()</code>
      </td>
      <td>
        retrieve user information (including their name) for a given user ID
      </td>
    </tr>
    <tr>
      <td>
        <code>getgrgid()</code>
      </td>
      <td>
        retrieve group information (including its name) for a given group ID
      </td>
    </tr>
    <tr>
      <td>
        Sorting
      </td>
      <td>
        <code>qsort()</code>
      </td>
      <td>
        quick-sort an array
      </td>
    </tr>
  </tbody>
</table>

## Frequently Asked Questions (FAQ)

**Question 1.** Do we still meet the limit of the 8-byte length of the user/group ids? Our user/group ids are longer than 8 bytes.

**Answer.** Yes. You can fit the 8-byte limit by format specifier for a printf()-like function. Please refer to below:
```c
printf("output: %8.8s:%8.8s\n", "uidlonglong", "gidlonglong");
```
Output:
```bash
uidlongl:gidlongl
```

**Question 2.** '$ ./gentree.sh demo.tree' command does not work!
```bash
Generating tree from 'tools/demo.tree'...
Failed to create named pipe './demo/subdir3/pipe'.
Failed to create unix socket './demo/subdir3/socket'.
Done. Generated 4 files, 2 links, 0 fifos, and 0 sockets. 2 errors reported.
```

**Answer.** Please delete the previously created directory and run ./gentree.sh demo.tree again.
```bash
$ rm -rf demo/
$ ./gentree.sh demo.tree
```

**Question 3.** Does the 54-byte limit apply to the indentation in the left side of the file/dir name? 

**Answer.** Yes. The 54-byte limit includes the indentation space in the left side of the name.

**Question 4.** What's the maximum depth of the nested directories that we should support?  

**Answer.** We recommend supporting the depth of up to 21. 20 x (2-byte space per each step) = 40 bytes. So, the file name in the directory depth of 21 would have 14-byte remaining space to represent. You can print out an error if the depth level is over 21 or continue to support a larger depth until you can't represent any file name. It's up to you, but we won't test any depth over 21.

**Question 5.** When I checked the behavior in the PowerPoint, I understood that the handling of path and name overflow occurs only when the -v option is provided. However, it seems that the reference/dirtree executable performs this action even without the -v option.
How would it be appropriate to implement this behavior?

**Answer.** I have written a binary that covers all cases. However, for the convenience of students, I have specified only the essential cases so that they do not have to cover all cases. That is, path and name overflow is graded only when -v option is present. Therefore, it does not matter whether path and name overflow is handled in all cases or only when -v option is present.

**Question 6.** What is the maximum path length limit for the target directory?

**Answer.** You don't need to handle overflow in target directory.

**Question 7.** Should we handle overflow in boundary condition?

**Answer.** You don't have to worry too much about boundary conditions for overflow handling, we'll just grade on output that's long enough.

<div align="center" style="font-size: 1.75em;">

**Happy coding!**
</div>
