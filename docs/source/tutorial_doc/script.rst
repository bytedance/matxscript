.. script

#############################################
Script
#############################################

********************************************
Functionality
********************************************
matx.script is used to compile Python function/class into corresponding C++ function/class. 

********************************************
Usage
********************************************
Note that we enforce type annotation of the class members and signatures of the functions.

********************************************
Example 1
********************************************
.. code-block:: python3 

   import matx

   # make a class and compile it as a operator
   class foo:
      
      def __init__(self, i:int)->None: # annotation of the function signature
         self._i:int = i # annotation of class member
      
      def add(self, j:int)->int:
         print("going to return self._i+j")
         return self._i+j
         
      def hello(self)->None:
         print("hello world")

   obj = matx.script(foo)(1)
   rc = obj.add(2)

********************************************
Example 2
********************************************

.. code-block:: python3 

   import matx

   # make a function and compile it as a operator
   @matx.script
   def boo(a:str, b:str)->str:  # annotation of the function signature
      return a+b
   
   rc = boo("ab", "cd")

