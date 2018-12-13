#!/usr/bin/expect -f
# Filename: beanstalk-purge
# http://grahamc.com/blog/clear-all-beanstalk-jobs-in-a-tube
# Use: beanstalk-purge <host> <port> <tube> <count>
# Ex: beanstalk-purge 127.0.0.1 11300 snitch.site 35000
set timeout 1

spawn telnet [lindex $argv 0] [lindex $argv 1]
sleep 1
send "use [lindex $argv 2]\n"
expect "USING"

for {set i 1} {$i < [lindex $argv 3]} { incr i 1 } {
    send_user "Proccessing $i\n"
    expect -re {.*} {}
    send "peek-ready\n"
    expect -re {FOUND (\d*) \d*}
    send "delete $expect_out(1,string)\n"
    expect "DELETED"
}
