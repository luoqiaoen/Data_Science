/* From cd.bookings cd.facilities and cd.members database*/

/* All information from cd.facilities*/
SELECT * from cd.facilities

/* print all of the facilities and their cost to members*/
SELECT name, membercost FROM cd.facilities

/* a list of facilities that charge member fee*/
SELECT * FROM cd.facilities
WHERE membercost > 0

/* select facilities that charge member fee, and the fee is less than
1/50th of the monthly maintenance cost*/
SELECT facid, name, membercost, monthlymaintenance FROM cd.facilities
WHERE membercost < monthlymaintenance/50.0 AND membercost > 0

/* produce a list of facilities with "Tennis" in their names*/
SELECT * FROM cd.facilities
WHERE name LIKE '%Tennis%'

/* details of facilities with ID 1 and 5 without using OR*/
SELECT * FROM cd.facilities
WHERE facid in (1,5)

/* list of members who joined after 09/2012*/
SELECT * FROM cd.members
WHERE joindate >= '2012-09-01'

/* ordered list of the first 10 surnames in member*/
SELECT DISTINCT surname FROM cd.members
ORDER BY surname
LIMIT 10

/* get signup date of your last member*/
SELECT joindate FROM cd.members
ORDER BY joindate DESC
LIMIT 1

/* get signup date of your last member*/
SELECT COUNT(*) FROM cd.facilities
WHERE guestcost >= 10

/* list of total number of slots booked per facilitiy in the month of 2012/09
produce and output table with facid, slots, sorted by number of slots*/
SELECT facid, SUM(slots) as "Total Slots" from cd.bookings
WHERE starttime >= '2012-09-01' and starttime < '2012-10-01'
GROUP by facid
ORDER by SUM(slots)

/* list of facilities with more than 1000 slots booked, produce an output table consisting of facid and total slots, sorted by facility id*/
SELECT facid, SUM(slots) as "Total Slots" from cd.bookings
GROUP by facid
HAVING sum(slots) > 1000
ORDER by facid

/* list of start times for booking for tennis courts, for the date "2019-09-21"*/
SELECT bks.starttime as start, facs.name as name from cd.facilities facs INNER JOIN cd.bookings bks on facs.facid = bks.facid
WHERE facs.facid in (0,1) AND bks.starttime <= '2012-09-21' and bks.starttime <'2012-09-22'
ORDER by bks.starttime

/* list of the start times for bookings by memebers named "David Farrell"*/
SELECT bks.starttime from cd.bookings bks INNER JOIN cd.members mems ON mems.memid = bks.memid
WHERE mems.firstname = 'David' and mems.surname = 'Farrell'
