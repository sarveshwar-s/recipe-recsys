Food Recipe RecSys

## Routes

### "/" :
* Returns Trending recipes if the user is not logged in 
* Returns Personalized recipes if the user is logged into the system

### /items/<item_number>:
* Returns the description about the passed item number 
* Secondly, it also returns the items similar to the selected one. 

### /items/popular/<item_number>:
* Returns the description about the item selected from the list of trending recipes.
* Secondly, it also returns the items similar to the selected popular item 

### /items/reinforcement/<item_number>:
* Returns the description about the item selected from the list of recipes recommended by reinforcement algorithms
* Secondly, it also computes the items simmilar to the selected items

### /from_fridge/<user_id>:
* Returns the items filtered based on the ingredients available with the user

### /developer_api
* Returns live public api of this system
