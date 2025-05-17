Imagine that you want to build a new data center or factory or whatever. 
You need something like 600 MW (configurable) of power to run your new facility, with high reliability.
And you would like it to be low carbon. 

For the sake of this simple case, let's assume you have two options:
1. build a new solar+storage plant that will provide power most of the time, or
2. build a nuclear plant that will run continuously aside from refueling outages and other maintenance. 

Which plant will be most cost effective?
Are the construction times acceptable?
Is the reliability pattern acceptable?
Play around with this app to find out. 

## Simplifying Assumptions

There are lots of simplifying assumptions in this model. See the [source code](https://github.com/phines/compare-power-plants) for details. 
Here are a few of the more important ones:
- The model only evaluates costs over a 40 year period and ignores end-of-life value.
- The model ignores the costs and benefits of grid interconnection. Clearly the results would be different, and probably lean more toward the solar option, if one could affordably interconnect with a utility, sell excess power and buy power when needed.
- I somewhat randomly chose to place the plant in Nevada. You may select from a number of solar sites (with data from NREL) for your plant.

## Credits and Disclaimers

- Author: [Paul Hines](https://www.linkedin.com/in/paul-hines-energy/)
- Original [LinkedIn post](https://www.linkedin.com/posts/paul-hines-energy_compare-the-costs-of-powering-a-large-load-with-solar-activity-7110000000000000000/)
- Source code: [github.com/phines/compare-power-plants](https://github.com/phines/compare-power-plants)
- Caveat: This is a personal project; do not blame my employer(s) for any errors or shortcomings. 