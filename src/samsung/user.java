/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package samsung;

import weka.core.Instances;

/**
 *
 * @author Yang
 */
public class user {
    private Integer userId;
    private Instances instances;
    
    public user(Integer userid, Instances instances){
        this.userId = userid;
        this.instances = instances;
    }

    /**
     * @return the userId
     */
    public Integer getUserId() {
        return userId;
    }

    /**
     * @param userId the userId to set
     */
    public void setUserId(Integer userId) {
        this.userId = userId;
    }

    /**
     * @return the instances
     */
    public Instances getInstances() {
        return instances;
    }

    /**
     * @param instances the instances to set
     */
    public void setInstances(Instances instances) {
        this.instances = instances;
    }
    
    public static void main(String args[]) {
        
    }
    
}
